/* ref_hy4.cpp -- the host reference of the Tencent Hy4-flash plugin's new ops (docs/OPS.md,
 * "Hy4-flash"): independent hyper-connections, absorbed MLA with a sink, non-absorbed attention
 * with a sink, the compressed indexer's pooling and score, and Engram.
 *
 * The semantics are the design notes' (hy4flash/kernels/*.md), checked against Hugging Face
 * transformers at c2ec8505 (HYV4Flash*). Conventions of this file, which are libref's:
 *
 *   * f32 accumulation in a fixed order (keys ascending, then columns ascending), no FMA
 *     (-ffp-contract=off), every output rounded ONCE to its dtype. A sum of squares and a dot
 *     product the notes call out as double are summed in double.
 *   * Scores and softmax are f32: HF rounds scores to bf16 before the softmax, and that is not
 *     reproduced (plan D4); cos/sin are f32 (plan D5).
 *   * One thread owns whole outputs (OpenMP over rows or (row, head)), no reduction clause, so
 *     results do not depend on the thread count.
 *   * A call with no rows is RAD_E_SHAPE, an absent required operand RAD_E_INVAL, a dtype the
 *     reference cannot read RAD_E_DTYPE (ref_registry.cpp), as everywhere in libref.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace ref;

namespace {

/* [rows, heads*width] read as [rows, heads, width], when it is declared the narrow way. */
void heads_view(RadTensor& t, int64_t h, int64_t width) {
    if (h > 0 && width > 0 && t.rank == 2 && t.shape[1] == h * width) {
        t.rank = 3;
        t.shape[1] = h;
        t.shape[2] = width;
        t.stride[2] = t.stride[1];
        t.stride[1] *= width;
    }
}

bool matrix(const RadTensor& t, int64_t rows, int64_t cols) {
    return t.data && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}

int64_t ld_idx(const RadTensor* t, int64_t i) { return ld_int(t->dtype, t->data, offlin(t, i)); }

/* Python's floor modulus, torch.remainder: non-negative for a positive modulus. */
int64_t floor_mod(int64_t a, int64_t b) {
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}
/* torch's int64 multiply wraps; C++'s is undefined on overflow. */
int64_t wrap_mul(int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); }

float sign_of(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

}  // namespace

/* ================================================================== ihc_pre / ihc_post */

/* Independent hyper-connection read (HYV4FlashHyperConnection and HyperHead).
 *   f = f32(h);  r = rsqrt(sum f^2 / (hc n) + eps);  c[j] = (fn[j] . f) r
 *   pre[j]  = sigmoid(c[j] scale[0] + base[j]) + hc_eps               j < hc
 *   x       = round(sum_j pre[j] f[j])
 *   post[j] = post_mag sigmoid(c[hc+j] scale[1] + base[hc+j]) + hc_eps   (when `post` is present) */
int ref_ihc_pre(const RadArgs* a, RadStream) {
    if (a->n_t < 5 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4))
        return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3], &x = a->t[4];
    const RadTensor* post = a->n_t > 5 ? t_in(a, 5) : nullptr;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0), rows = p_int(a, "rows", 0);
    const float eps = p_f32(a, "eps", 0), heps = p_f32(a, "hc_eps", 0);
    const int64_t width = hc * n;
    if (n < 1 || hc < 1 || hc > 16 || rows != (post ? 2 * hc : hc) || !std::isfinite(eps) ||
        eps <= 0 || !std::isfinite(heps) || heps < 0)
        return RAD_E_SHAPE;
    float pmag = 0;
    if (post) {
        if (!p_find(a, "post_mag")) return RAD_E_INVAL;
        pmag = p_f32(a, "post_mag", 0);
        if (!std::isfinite(pmag)) return RAD_E_SHAPE;
    }
    if (h.rank != 2 || h.shape[0] < 1) return RAD_E_SHAPE;
    const int64_t M = h.shape[0];
    if (!matrix(h, M, width) || !matrix(fn, rows, width) || scale.rank != 1 ||
        numel(&scale) != (post ? 2 : 1) || base.rank != 1 || numel(&base) != rows ||
        !matrix(x, M, n) || (post && !matrix(*post, M, hc)))
        return RAD_E_SHAPE;
    if (post && post->dtype != RAD_F32) return RAD_E_DTYPE;
    const float s0 = ldt(&scale, 0), s1 = post ? ldt(&scale, laststride(&scale)) : 0.0f;
    REF_PAR_FOR(M * (rows + 1) * width)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        double sq = 0;
        for (int64_t i = 0; i < width; ++i) {
            const double v = ldt(&h, hb + i * hs);
            sq += v * v;
        }
        const float ms = (float)(sq / (double)width);
        const float inv = 1.0f / std::sqrt(ms + eps);
        float c[32], pre[16];
        for (int64_t j = 0; j < rows; ++j) {
            const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
            float dot = 0;
            for (int64_t i = 0; i < width; ++i) dot += ldt(&fn, fb + i * fs) * ldt(&h, hb + i * hs);
            c[j] = dot * inv;
        }
        for (int64_t j = 0; j < hc; ++j)
            pre[j] = act_sigmoid(c[j] * s0 + ldt(&base, j * laststride(&base))) + heps;
        if (post)
            for (int64_t j = 0; j < hc; ++j)
                stt(post, rowoff(post, r, hc) + j * laststride(post),
                    pmag * act_sigmoid(c[hc + j] * s1 + ldt(&base, (hc + j) * laststride(&base))) +
                        heps);
        const int64_t xb = rowoff(&x, r, n), xs = laststride(&x);
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j) v += pre[j] * ldt(&h, hb + (j * n + i) * hs);
            stt(&x, xb + i * xs, v);
        }
    }
    return RAD_OK;
}

/* h[j][i] = round(post[j] * f32(y[i]) + f32(h[j][i])), in place. */
int ref_ihc_post(const RadArgs* a, RadStream) {
    if (a->n_t < 3 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2)) return RAD_E_INVAL;
    const RadTensor &y = a->t[0], &post = a->t[1], &h = a->t[2];
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n < 1 || hc < 1 || h.rank != 2 || h.shape[0] < 1) return RAD_E_SHAPE;
    const int64_t M = h.shape[0];
    if (!matrix(h, M, hc * n) || !matrix(y, M, n) || !matrix(post, M, hc)) return RAD_E_SHAPE;
    if (post.dtype != RAD_F32) return RAD_E_DTYPE;
    REF_PAR_FOR(M * hc * n)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t hb = rowoff(&h, r, hc * n), hs = laststride(&h);
        const int64_t yb = rowoff(&y, r, n), ys = laststride(&y);
        for (int64_t j = 0; j < hc; ++j) {
            const float p = ldt(&post, rowoff(&post, r, hc) + j * laststride(&post));
            for (int64_t i = 0; i < n; ++i) {
                const float prod = p * ldt(&y, yb + i * ys);
                stt(&h, hb + (j * n + i) * hs, prod + ldt(&h, hb + (j * n + i) * hs));
            }
        }
    }
    return RAD_OK;
}

/* ================================================================== mla_project_kv */

/* The checkpoint's kv_b read per head: head h's rows start at h (k_dim + v_dim), its k_dim key rows
 * first, then its v_dim value rows.
 *   absorb: y[m,h,j] = sum_{c<k_dim} a[m,h,c] b[h(kd+vd) + c][j]          [M,H,kd] -> [M,H,L]
 *   expand: y[m,h,c] = sum_{j<L}     a[m,h,j] b[h(kd+vd) + kd + c][j]     [M,H,L]  -> [M,H,vd] */
int ref_mla_project_kv(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    RadTensor x = a->t[0], y = a->t[2];
    const RadTensor& w = a->t[1];
    const int64_t H = p_int(a, "heads", 0), kd = p_int(a, "k_dim", 0), vd = p_int(a, "v_dim", 0),
                  L = p_int(a, "latent_dim", 0);
    const char* mode = p_str(a, "mode", "");
    const bool absorb = !std::strcmp(mode, "absorb"), expand = !std::strcmp(mode, "expand");
    if ((!absorb && !expand) || H < 1 || kd < 1 || vd < 1 || L < 1) return RAD_E_SHAPE;
    const int64_t in_w = absorb ? kd : L, out_w = absorb ? L : vd;
    heads_view(x, H, in_w);
    heads_view(y, H, out_w);
    if (x.rank != 3 || y.rank != 3 || x.shape[0] < 1 || x.shape[1] != H || x.shape[2] != in_w ||
        w.rank != 2 || w.shape[0] != H * (kd + vd) || w.shape[1] != L ||
        y.shape[0] != x.shape[0] || y.shape[1] != H || y.shape[2] != out_w)
        return RAD_E_SHAPE;
    const int64_t M = x.shape[0];
    REF_PAR_FOR(M * H * out_w * in_w)
    for (int64_t job = 0; job < M * H; ++job) {
        const int64_t r = job / H, head = job % H;
        for (int64_t col = 0; col < out_w; ++col) {
            float sum = 0;
            for (int64_t j = 0; j < in_w; ++j) {
                const int64_t wr = head * (kd + vd) + (absorb ? j : kd + col), wc = absorb ? col : j;
                sum += ldt(&x, r * x.stride[0] + head * x.stride[1] + j * x.stride[2]) *
                       ldt(&w, wr * w.stride[0] + wc * w.stride[1]);
            }
            stt(&y, r * y.stride[0] + head * y.stride[1] + col * y.stride[2], sum);
        }
    }
    return RAD_OK;
}

/* ================================================================== latent attention with a sink */

namespace {

/* One (query row, head): the keys of `[lo, hi)` read through page table row `bt_row`, softmax with
 * the sink joining the denominator once, value = the latent. No mapped key in range: zero. */
void mla_sink_one(const RadTensor& q, const RadTensor& qr, const RadTensor& kv, const RadTensor& bt,
                  int64_t bt_row, int64_t bs, int64_t lo, int64_t hi, float scale, float sink,
                  int64_t r, int64_t head, int64_t L, int64_t R, const RadTensor& o) {
    std::vector<int64_t> base;
    std::vector<float> s;
    base.reserve((size_t)std::max<int64_t>(hi - lo, 0));
    for (int64_t t = lo; t < hi; ++t) {
        const int64_t page = ld_int(bt.dtype, bt.data, bt_row * bt.stride[0] + (t / bs) * bt.stride[1]);
        if (page < 0 || page >= kv.shape[0]) continue;
        const int64_t b = page * kv.stride[0] + (t % bs) * kv.stride[2];
        float v = 0;
        for (int64_t c = 0; c < L; ++c)
            v += ldt(&q, r * q.stride[0] + head * q.stride[1] + c * q.stride[2]) *
                 ldt(&kv, b + c * kv.stride[3]);
        for (int64_t c = 0; c < R; ++c)
            v += ldt(&qr, r * qr.stride[0] + head * qr.stride[1] + c * qr.stride[2]) *
                 ldt(&kv, b + (L + c) * kv.stride[3]);
        base.push_back(b);
        s.push_back(v * scale);
    }
    const int64_t ob = r * o.stride[0] + head * o.stride[1];
    if (s.empty()) {
        for (int64_t c = 0; c < L; ++c) stt(&o, ob + c * o.stride[2], 0.0f);
        return;
    }
    /* The maximum includes the sink (HF takes it over the concatenated logits); it only moves the
     * shift, never the value. */
    float mx = sink;
    for (float v : s) mx = std::max(mx, v);
    float den = 0;
    std::vector<float> val((size_t)L, 0.0f), w(s.size());
    for (size_t j = 0; j < s.size(); ++j) {
        w[j] = ref_exp(s[j] - mx);
        den += w[j];
        for (int64_t c = 0; c < L; ++c) val[(size_t)c] += w[j] * ldt(&kv, base[j] + c * kv.stride[3]);
    }
    den += ref_exp(sink - mx);
    for (int64_t c = 0; c < L; ++c) stt(&o, ob + c * o.stride[2], val[(size_t)c] / den);
}

bool sink_vector_ok(const RadTensor& sinks, int64_t H) { return sinks.rank == 1 && numel(&sinks) >= H; }

}  // namespace

/* mla_attn_paged with a window and a sink. Sequence s owns query rows [cu[s], cu[s+1]), its
 * context after the step is seqused[s], row i sits at p = seqused[s] - qlen + i and sees keys j in
 * [max(0, p - window + 1), p] (window 0: from 0). The window counts the query's own token. */
int ref_mla_attn_paged_sink(const RadArgs* a, RadStream) {
    if (a->n_t < 8 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4) ||
        !t_in(a, 5) || !t_in(a, 6) || !t_in(a, 7))
        return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[7];
    const RadTensor &kv = a->t[2], &bt = a->t[3], &su = a->t[4], &cu = a->t[5], &sinks = a->t[6];
    const int64_t H = p_int(a, "heads", 0), L = p_int(a, "latent_dim", 0),
                  R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0),
                  window = p_int(a, "window", 0);
    const float scale = p_f32(a, "scale", 0);
    heads_view(q, H, L);
    heads_view(qr, H, R);
    heads_view(o, H, L);
    if (H < 1 || L < 1 || R < 1 || bs < 1 || window < 0 || !std::isfinite(scale) || scale <= 0 ||
        q.rank != 3 || q.shape[0] < 1 || q.shape[1] != H || q.shape[2] != L || qr.rank != 3 ||
        qr.shape[0] != q.shape[0] || qr.shape[1] != H || qr.shape[2] != R || kv.rank != 4 ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R || bt.rank != 2 ||
        su.rank != 1 || cu.rank != 1 || numel(&cu) != numel(&su) + 1 || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != H || o.shape[2] != L || !sink_vector_ok(sinks, H))
        return RAD_E_SHAPE;
    const int64_t n_seq = numel(&su), rows = q.shape[0];
    if (cu_at(&cu, 0) != 0 || cu_at(&cu, n_seq) != rows || bt.shape[0] < n_seq) return RAD_E_SHAPE;
    REF_PAR_FOR_DYN(rows * H * 64)
    for (int64_t job = 0; job < rows * H; ++job) {
        const int64_t r = job / H, head = job % H;
        const int64_t s = seq_of(&cu, n_seq, r);
        const int64_t qlen = cu_at(&cu, s + 1) - cu_at(&cu, s), i = r - cu_at(&cu, s);
        const int64_t ctx = ld_int(su.dtype, su.data, s * su.stride[0]);
        const int64_t pos = ctx - qlen + i;
        const int64_t hi = std::min(pos + 1, bt.shape[1] * bs);
        const int64_t lo = window > 0 ? std::max<int64_t>(0, pos - window + 1) : 0;
        mla_sink_one(q, qr, kv, bt, s, bs, lo, hi, scale,
                     ldt(&sinks, head * sinks.stride[0]), r, head, L, R, o);
    }
    return RAD_OK;
}

/* mla_attn's per-query page table with a rotary part and a sink: tokens [0, length[r]) of row r's
 * own table row `bt[r]`, page bt[r][t / block_size], slot t % block_size. */
int ref_mla_attn_sel_sink(const RadArgs* a, RadStream) {
    if (a->n_t < 7 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4) ||
        !t_in(a, 5) || !t_in(a, 6))
        return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[6];
    const RadTensor &kv = a->t[2], &bt = a->t[3], &len = a->t[4], &sinks = a->t[5];
    const int64_t H = p_int(a, "heads", 0), L = p_int(a, "latent_dim", 0),
                  R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0);
    const float scale = p_f32(a, "scale", 0);
    heads_view(q, H, L);
    heads_view(qr, H, R);
    heads_view(o, H, L);
    if (H < 1 || L < 1 || R < 1 || bs < 1 || !std::isfinite(scale) || scale <= 0 || q.rank != 3 ||
        q.shape[0] < 1 || q.shape[1] != H || q.shape[2] != L || qr.rank != 3 ||
        qr.shape[0] != q.shape[0] || qr.shape[1] != H || qr.shape[2] != R || kv.rank != 4 ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R || bt.rank != 2 ||
        bt.shape[0] < q.shape[0] || len.rank != 1 || len.shape[0] < q.shape[0] || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != H || o.shape[2] != L || !sink_vector_ok(sinks, H))
        return RAD_E_SHAPE;
    const int64_t rows = q.shape[0];
    REF_PAR_FOR_DYN(rows * H * 64)
    for (int64_t job = 0; job < rows * H; ++job) {
        const int64_t r = job / H, head = job % H;
        const int64_t hi = std::clamp(ld_int(len.dtype, len.data, r * len.stride[0]), (int64_t)0,
                                      bt.shape[1] * bs);
        mla_sink_one(q, qr, kv, bt, r, bs, 0, hi, scale, ldt(&sinks, head * sinks.stride[0]), r,
                     head, L, R, o);
    }
    return RAD_OK;
}

/* ================================================================== non-absorbed prefill */

/* kvb [N, H (kd+vd)] and k_rope [N, R] -> k [N, H, kd+R] (the rope part shared by every head) and
 * v [N, H, vd]. A copy; the dtypes of the operands may differ and are converted. */
int ref_mla_kv_split(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor &kvb = a->t[0], &kr = a->t[1];
    RadTensor k = a->t[2], v = a->t[3];
    const int64_t H = p_int(a, "heads", 0), kd = p_int(a, "k_dim", 0), vd = p_int(a, "v_dim", 0),
                  R = p_int(a, "rope_dim", 0);
    if (H < 1 || kd < 1 || vd < 1 || R < 1) return RAD_E_SHAPE;
    heads_view(k, H, kd + R);
    heads_view(v, H, vd);
    if (kvb.rank != 2 || kvb.shape[0] < 1 || kvb.shape[1] != H * (kd + vd) || kr.rank != 2 ||
        kr.shape[0] != kvb.shape[0] || kr.shape[1] != R || k.rank != 3 ||
        k.shape[0] != kvb.shape[0] || k.shape[1] != H || k.shape[2] != kd + R || v.rank != 3 ||
        v.shape[0] != kvb.shape[0] || v.shape[1] != H || v.shape[2] != vd)
        return RAD_E_SHAPE;
    const int64_t N = kvb.shape[0];
    REF_PAR_FOR(N * H * (kd + R + vd))
    for (int64_t i = 0; i < N; ++i)
        for (int64_t h = 0; h < H; ++h) {
            const int64_t src = i * kvb.stride[0] + h * (kd + vd) * kvb.stride[1];
            for (int64_t c = 0; c < kd; ++c)
                stt(&k, i * k.stride[0] + h * k.stride[1] + c * k.stride[2],
                    ldt(&kvb, src + c * kvb.stride[1]));
            for (int64_t c = 0; c < R; ++c)
                stt(&k, i * k.stride[0] + h * k.stride[1] + (kd + c) * k.stride[2],
                    ldt(&kr, i * kr.stride[0] + c * kr.stride[1]));
            for (int64_t c = 0; c < vd; ++c)
                stt(&v, i * v.stride[0] + h * v.stride[1] + c * v.stride[2],
                    ldt(&kvb, src + (kd + c) * kvb.stride[1]));
        }
    return RAD_OK;
}

/* Dense attention over a key prefix, with a window and a sink. Sequence s has m_s query rows
 * [cu_q[s], cu_q[s+1]) and n_s >= m_s key rows [cu_k[s], cu_k[s+1]); query i sits at key index
 * p = n_s - m_s + i and sees j <= p with p - j < window (window 0: all). */
int ref_attn_kv_sink(const RadArgs* a, RadStream) {
    if (a->n_t < 7 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4) ||
        !t_in(a, 5) || !t_in(a, 6))
        return RAD_E_INVAL;
    RadTensor q = a->t[0], k = a->t[1], v = a->t[2], o = a->t[6];
    const RadTensor &cuq = a->t[3], &cuk = a->t[4], &sinks = a->t[5];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0),
                  window = p_int(a, "window", 0);
    const float scale = p_f32(a, "scale", 0);
    if (H < 1 || D < 1 || window < 0 || !std::isfinite(scale) || scale <= 0) return RAD_E_SHAPE;
    heads_view(q, H, D);
    heads_view(k, H, D);
    if (v.rank == 2 && v.shape[1] % H == 0) heads_view(v, H, v.shape[1] / H);
    if (o.rank == 2 && o.shape[1] % H == 0) heads_view(o, H, o.shape[1] / H);
    if (q.rank != 3 || k.rank != 3 || v.rank != 3 || o.rank != 3 || q.shape[0] < 1 ||
        k.shape[0] < 1 || q.shape[1] != H || q.shape[2] != D || k.shape[1] != H || k.shape[2] != D ||
        v.shape[0] != k.shape[0] || v.shape[1] != H || o.shape[0] != q.shape[0] ||
        o.shape[1] != H || o.shape[2] != v.shape[2] || cuq.rank != 1 || cuk.rank != 1 ||
        numel(&cuq) != numel(&cuk) || numel(&cuq) < 2 || !sink_vector_ok(sinks, H))
        return RAD_E_SHAPE;
    const int64_t V = v.shape[2], n_seq = numel(&cuq) - 1, rows = q.shape[0];
    if (cu_at(&cuq, 0) != 0 || cu_at(&cuq, n_seq) != rows) return RAD_E_SHAPE;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t m = cu_at(&cuq, s + 1) - cu_at(&cuq, s);
        const int64_t n = cu_at(&cuk, s + 1) - cu_at(&cuk, s);
        if (m < 0 || n < m || cu_at(&cuk, s) < 0 || cu_at(&cuk, s + 1) > k.shape[0])
            return RAD_E_SHAPE;
    }
    REF_PAR_FOR_DYN(rows * H * 64)
    for (int64_t job = 0; job < rows * H; ++job) {
        const int64_t r = job / H, head = job % H;
        const int64_t s = seq_of(&cuq, n_seq, r);
        const int64_t m = cu_at(&cuq, s + 1) - cu_at(&cuq, s), n = cu_at(&cuk, s + 1) - cu_at(&cuk, s);
        const int64_t k0 = cu_at(&cuk, s), p = n - m + (r - cu_at(&cuq, s));
        const int64_t lo = window > 0 ? std::max<int64_t>(0, p - window + 1) : 0;
        std::vector<float> sc((size_t)(p - lo + 1));
        float mx = ldt(&sinks, head * sinks.stride[0]);
        const float sink = mx;
        for (int64_t j = lo; j <= p; ++j) {
            float dot = 0;
            for (int64_t c = 0; c < D; ++c)
                dot += ldt(&q, r * q.stride[0] + head * q.stride[1] + c * q.stride[2]) *
                       ldt(&k, (k0 + j) * k.stride[0] + head * k.stride[1] + c * k.stride[2]);
            sc[(size_t)(j - lo)] = dot * scale;
            mx = std::max(mx, sc[(size_t)(j - lo)]);
        }
        float den = 0;
        std::vector<float> val((size_t)V, 0.0f);
        for (int64_t j = lo; j <= p; ++j) {
            const float w = ref_exp(sc[(size_t)(j - lo)] - mx);
            den += w;
            for (int64_t c = 0; c < V; ++c)
                val[(size_t)c] +=
                    w * ldt(&v, (k0 + j) * v.stride[0] + head * v.stride[1] + c * v.stride[2]);
        }
        den += ref_exp(sink - mx);
        for (int64_t c = 0; c < V; ++c)
            stt(&o, r * o.stride[0] + head * o.stride[1] + c * o.stride[2],
                p >= lo ? val[(size_t)c] / den : 0.0f);
    }
    return RAD_OK;
}

/* ================================================================== gate_mul */

/* y = round(f32(a) * sigmoid(f32(g))): one rounding. act_mul's sigmoid form rounds the sigmoid to
 * y's dtype first (the unfused pair), which is a different bf16 result. */
int ref_gate_mul(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &g = a->t[1], &y = a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n < 1 || y.rank < 1) return RAD_E_SHAPE;
    const int64_t M = rows_of(&y, n);
    if (M < 1 || numel(&x) < M * n || numel(&g) < M * n || y.shape[y.rank - 1] != n)
        return RAD_E_SHAPE;
    REF_PAR_FOR(M * n)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t xb = rowoff(&x, r, n), gb = rowoff(&g, r, n), yb = rowoff(&y, r, n);
        for (int64_t i = 0; i < n; ++i)
            stt(&y, yb + i * laststride(&y),
                ldt(&x, xb + i * laststride(&x)) * act_sigmoid(ldt(&g, gb + i * laststride(&g))));
    }
    return RAD_OK;
}

/* ================================================================== the compressed indexer */

namespace {

/* Rotary angle component of pair `j` for mode imrope_gptj with sections (t, h, w): pair j takes the
 * h position at j % 3 == 1 below 3 * h, the w position at j % 3 == 2 below 3 * w, else t. */
int imrope_component(int64_t j, int64_t sec_h, int64_t sec_w) {
    if (j % 3 == 1 && j < 3 * sec_h) return 1;
    if (j % 3 == 2 && j < 3 * sec_w) return 2;
    return 0;
}

}  // namespace

/* Pools each closed group of `ratio` token keys into one rotated key (HYV4FlashCompressedIndexer.
 * pool_groups, then the rotary at the group's first position), all in f32 and rounded to bf16
 * twice as HF does: after the norm, and after the rotation. "Rounded" means to `k`'s dtype, as HF
 * rounds to the token keys' dtype (an f32 model rounds nothing). */
int ref_idx_pool_gated(const RadArgs* a, RadStream) {
    if (a->n_t < 6 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4) ||
        !t_in(a, 5))
        return RAD_E_INVAL;
    const RadTensor &k = a->t[0], &wv = a->t[1], &wg = a->t[2], &ape = a->t[3], &gain = a->t[4],
                    &pos = a->t[5];
    const RadTensor* page = a->n_t > 6 ? t_in(a, 6) : nullptr;
    const RadTensor* cache = a->n_t > 7 ? t_in(a, 7) : nullptr;
    const RadTensor* out = a->n_t > 8 ? t_in(a, 8) : nullptr;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), rd = p_int(a, "rotary_dim", 0);
    const float eps = p_f32(a, "eps", 0);
    const double theta = (double)p_f32(a, "theta", 0);
    const char* mode = p_str(a, "mode", "");
    const bool gptj = !std::strcmp(mode, "gptj"), imrope = !std::strcmp(mode, "imrope_gptj");
    if (n < 1 || ratio < 1 || ratio > 16 || rd < 2 || rd > n || (rd & 1) || !(eps > 0) ||
        !std::isfinite(eps) || !(theta > 1.0) || (!gptj && !imrope))
        return RAD_E_SHAPE;
    int64_t sec_h = 0, sec_w = 0;
    if (imrope) {
        const char* s = p_str(a, "sections", nullptr);
        long t = 0, h = 0, w = 0;
        if (!s || std::sscanf(s, "%ld %ld %ld", &t, &h, &w) != 3 || t < 0 || h < 0 || w < 0 ||
            t + h + w != rd / 2)
            return RAD_E_SHAPE;
        sec_h = h;
        sec_w = w;
    }
    if (!page != !cache) return RAD_E_INVAL;
    if (!cache && !out) return RAD_E_INVAL;
    if (k.rank != 2 || k.shape[0] < ratio || k.shape[0] % ratio || k.shape[1] != n)
        return RAD_E_SHAPE;
    const int64_t W = k.shape[0] / ratio;
    if (!matrix(wv, n, n) || !matrix(wg, n, n) || !matrix(ape, ratio, n) || gain.rank != 1 ||
        numel(&gain) != n || pos.rank != (imrope ? 2 : 1) ||
        (imrope ? (pos.shape[0] != 3 || pos.shape[1] < W) : pos.shape[0] < W))
        return RAD_E_SHAPE;
    if (page && (page->rank != 1 || page->shape[0] < W)) return RAD_E_SHAPE;
    if (cache && (cache->rank != 4 || cache->shape[0] < 1 || cache->shape[1] != 1 ||
                  cache->shape[2] * cache->shape[3] != n || cache->stride[3] != 1 ||
                  cache->stride[2] != cache->shape[3] || cache->stride[0] < n))
        return RAD_E_SHAPE;
    if (out && !matrix(*out, W, n)) return RAD_E_SHAPE;
    /* Pages are written by distinct groups; two groups naming one page is a caller bug that would
     * make the result depend on the thread order. */
    if (page)
        for (int64_t w = 0; w < W; ++w) {
            const int64_t pw = ld_idx(page, w);
            if (pw < 0 || pw >= cache->shape[0]) continue;
            for (int64_t u = 0; u < w; ++u)
                if (ld_idx(page, u) == pw) return RAD_E_INVAL;
        }
    REF_PAR_FOR_DYN(W * 2 * n * n * ratio)
    for (int64_t w = 0; w < W; ++w) {
        std::vector<float> val((size_t)(ratio * n)), gat((size_t)(ratio * n)), y((size_t)n);
        for (int64_t s = 0; s < ratio; ++s) {
            const int64_t kb = rowoff(&k, w * ratio + s, n), ks = laststride(&k);
            for (int64_t c = 0; c < n; ++c) {
                float dv = 0, dg = 0;
                for (int64_t j = 0; j < n; ++j) {
                    const float xj = ldt(&k, kb + j * ks);
                    dv += ldt(&wv, c * wv.stride[0] + j * wv.stride[1]) * xj;
                    dg += ldt(&wg, c * wg.stride[0] + j * wg.stride[1]) * xj;
                }
                val[(size_t)(s * n + c)] = dv;
                gat[(size_t)(s * n + c)] = dg + ldt(&ape, s * ape.stride[0] + c * ape.stride[1]);
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            float mx = -INFINITY;
            for (int64_t s = 0; s < ratio; ++s) mx = std::max(mx, gat[(size_t)(s * n + c)]);
            float e[16], den = 0;
            for (int64_t s = 0; s < ratio; ++s) {
                e[s] = ref_exp(gat[(size_t)(s * n + c)] - mx);
                den += e[s];
            }
            float acc = 0;
            for (int64_t s = 0; s < ratio; ++s) acc += val[(size_t)(s * n + c)] * (e[s] / den);
            y[(size_t)c] = acc;
        }
        double sq = 0;
        for (int64_t c = 0; c < n; ++c) sq += (double)y[(size_t)c] * (double)y[(size_t)c];
        const float inv = 1.0f / std::sqrt((float)(sq / (double)n) + eps);
        std::vector<float> z((size_t)n);
        for (int64_t c = 0; c < n; ++c)
            z[(size_t)c] = round_dt(k.dtype, ldt(&gain, c * gain.stride[0]) * (y[(size_t)c] * inv));
        for (int64_t i = 0; i < rd / 2; ++i) {
            const int comp = imrope ? imrope_component(i, sec_h, sec_w) : 0;
            const int64_t pv = imrope ? ld_int(pos.dtype, pos.data, comp * pos.stride[0] + w * pos.stride[1])
                                      : ld_int(pos.dtype, pos.data, w * pos.stride[0]);
            const float inv_freq = (float)(1.0 / std::pow(theta, (double)(2 * i) / (double)rd));
            const float ang = (float)pv * inv_freq;
            const float cs = ref_cos(ang), sn = ref_sin(ang);
            const int64_t o0 = n - rd + 2 * i;
            const float x0 = z[(size_t)o0], x1 = z[(size_t)o0 + 1];
            z[(size_t)o0] = round_dt(k.dtype, x0 * cs - x1 * sn);
            z[(size_t)o0 + 1] = round_dt(k.dtype, x1 * cs + x0 * sn);
        }
        if (out)
            for (int64_t c = 0; c < n; ++c)
                stt(out, rowoff(out, w, n) + c * laststride(out), z[(size_t)c]);
        if (cache) {
            const int64_t pg = ld_idx(page, w);
            if (pg >= 0 && pg < cache->shape[0])
                for (int64_t c = 0; c < n; ++c) stt(cache, pg * cache->stride[0] + c, z[(size_t)c]);
        }
    }
    return RAD_OK;
}

/* score[m][b] = sum_h w_h relu(q[m][h] . key_b), w_h = gate[m][h] (heads n)^-0.5, summed over the
 * first and the second half of the heads separately and then added. Keys b < nc[seq] with a valid
 * page; every other column -inf. */
int ref_idx_score(const RadArgs* a, RadStream) {
    if (a->n_t < 7 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4) ||
        !t_in(a, 5) || !t_in(a, 6))
        return RAD_E_INVAL;
    RadTensor q = a->t[0];
    const RadTensor &gate = a->t[1], &key = a->t[2], &bt = a->t[3], &cu = a->t[4], &nc = a->t[5],
                    &score = a->t[6];
    const int64_t n = p_int(a, "n", 0), H = p_int(a, "heads", 0);
    if (n < 1 || H < 1) return RAD_E_SHAPE;
    heads_view(q, H, n);
    if (q.rank != 3 || q.shape[0] < 1 || q.shape[1] != H || q.shape[2] != n || gate.rank != 2 ||
        gate.shape[0] != q.shape[0] || gate.shape[1] < H || bt.rank != 2 || cu.rank != 1 ||
        cu.shape[0] < 2 || score.rank != 2 || score.shape[0] != q.shape[0] || score.shape[1] < 1)
        return RAD_E_SHAPE;
    const bool paged = key.rank == 4;
    if (paged ? (key.shape[1] != 1 || key.shape[2] * key.shape[3] != n || key.stride[3] != 1 ||
                 key.stride[2] != key.shape[3] || key.stride[0] < n)
              : (key.rank != 2 || key.shape[1] != n))
        return RAD_E_SHAPE;
    const int64_t M = q.shape[0], n_seq = cu.shape[0] - 1, pages = key.shape[0];
    if (bt.shape[0] < n_seq || numel(&nc) < n_seq || cu_at(&cu, 0) != 0 || cu_at(&cu, n_seq) != M)
        return RAD_E_SHAPE;
    const float wscale = (float)(std::pow((double)H, -0.5) * std::pow((double)n, -0.5));
    const int64_t cols = std::min(score.shape[1], bt.shape[1]);
    REF_PAR_FOR(M * cols * H * n)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_of(&cu, n_seq, m);
        const int64_t nlive = std::min(ld_idx(&nc, s), cols);
        const int64_t half = H / 2;
        std::vector<float> wh((size_t)H);
        for (int64_t h = 0; h < H; ++h) wh[(size_t)h] = ldt(&gate, m * gate.stride[0] + h * gate.stride[1]) * wscale;
        for (int64_t b = 0; b < score.shape[1]; ++b) {
            float out = -INFINITY;
            const int64_t page = b < nlive ? ld_int(bt.dtype, bt.data, s * bt.stride[0] + b * bt.stride[1]) : -1;
            if (b < cols && page >= 0 && page < pages) {
                const int64_t kb = page * key.stride[0], ks = paged ? 1 : key.stride[1];
                float lo = 0, hi = 0;
                for (int64_t h = 0; h < H; ++h) {
                    float d = 0;
                    for (int64_t c = 0; c < n; ++c)
                        d += ldt(&q, m * q.stride[0] + h * q.stride[1] + c * q.stride[2]) *
                             ldt(&key, kb + c * ks);
                    const float term = wh[(size_t)h] * std::max(d, 0.0f);
                    (h < half ? lo : hi) += term;
                }
                out = lo + hi;
            }
            stt(&score, m * score.stride[0] + b * score.stride[1], out);
        }
    }
    return RAD_OK;
}

/* ================================================================== Engram */

/* The hashed n-gram ids of one step. Token k places back is the sequence's own token, or the pad id
 * before the document start (pos < k); ids at or past the token map read as pad:
 *   c_k = map[t_k < map_size ? t_k : pad];  m_k = c_k * mult[k] (int64, wrapping)
 *   mix_g = m_0 ^ ... ^ m_{g-1}  (g = 2 .. ngram);  head i of order g: floor_mod(mix_g, vocab[i]) + off[i]
 * `state` carries the committed ids before the step as ple_conv's window (ngram_ids' contract). */
int ref_engram_ids(const RadArgs* a, RadStream) {
    if (a->n_t < 12 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 2) || !t_in(a, 3) || !t_in(a, 4) ||
        !t_in(a, 5) || !t_in(a, 6) || !t_in(a, 7) || !t_in(a, 10) || !t_in(a, 11))
        return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &st = a->t[1], &cu = a->t[2], &pos = a->t[3], &map = a->t[4],
                    &mult = a->t[5], &vsz = a->t[6], &off = a->t[7];
    const RadTensor* nacc = t_in(a, 8);
    const RadTensor* sidx = t_in(a, 9);
    const RadTensor &ids = a->t[10], &mask = a->t[11];
    const int64_t heads = p_int(a, "heads", 0), ngram = p_int(a, "ngram", 0),
                  pad = p_int(a, "pad", -1), msize = p_int(a, "map_size", 0),
                  image = p_int(a, "image", -1), video = p_int(a, "video", -1);
    if (heads < 1 || ngram < 2 || ngram > 8 || heads % (ngram - 1) || msize < 1) return RAD_E_SHAPE;
    if (pad < 0 || pad >= msize) return RAD_E_INVAL;
    const int64_t per = heads / (ngram - 1), ctx = ngram - 1;
    if (tok.rank < 1 || numel(&tok) < 1 || numel(&map) < msize || numel(&mult) < ngram ||
        numel(&vsz) < heads || numel(&off) < heads || cu.rank != 1 || cu.shape[0] < 2)
        return RAD_E_SHAPE;
    const int64_t M = numel(&tok), n_seq = cu.shape[0] - 1;
    if (numel(&pos) < M || ids.rank != 2 || ids.shape[0] < M || ids.shape[1] != heads ||
        numel(&mask) < M || st.rank != 2 || st.shape[1] < ctx)
        return RAD_E_SHAPE;
    for (int64_t h = 0; h < heads; ++h)
        if (ld_idx(&vsz, h) <= 0 || ld_idx(&off, h) < 0) return RAD_E_SHAPE;
    const int64_t W = st.shape[1];
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t o = nacc ? ld_idx(nacc, s) - 1 : 0;
        const int64_t b0 = cu_at(&cu, s), b1 = cu_at(&cu, s + 1);
        if (o < 0) return RAD_E_INVAL;
        if (o + ctx > W || b0 < 0 || b1 < b0 || b1 > M) return RAD_E_SHAPE;
        if (nacc && b1 > b0 && ctx - 1 + (b1 - b0) > W) return RAD_E_SHAPE;
        const int64_t slot = sidx ? ld_idx(sidx, s) : s;
        if (slot >= st.shape[0]) return RAD_E_SHAPE;
    }
    const int64_t map_pad = ld_idx(&map, pad);
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t bos = cu_at(&cu, s), T = cu_at(&cu, s + 1) - bos;
        if (T <= 0) continue;
        const int64_t slot = sidx ? ld_idx(sidx, s) : s;
        if (slot < 0) continue;
        const int64_t o = nacc ? ld_idx(nacc, s) - 1 : 0;
        const int64_t sb = rowoff(&st, slot, W);
        int64_t prev[8];
        for (int64_t i = 0; i < ctx; ++i) prev[i] = ld_int(st.dtype, st.data, sb + o + i);
        /* The token `k` places before row t of the sequence; history index ctx + t - k. */
        auto at = [&](int64_t t, int64_t k) -> int64_t {
            const int64_t p = ctx + t - k;
            return p < ctx ? prev[p] : ld_idx(&tok, bos + p - ctx);
        };
        REF_PAR_FOR(T * heads * 4)
        for (int64_t t = 0; t < T; ++t) {
            const int64_t m = bos + t, pm = ld_idx(&pos, m);
            int64_t mixed[8];
            for (int64_t k = 0; k < ngram; ++k) {
                int64_t c = map_pad;
                if (pm >= k) {
                    const int64_t tk = at(t, k);
                    if (tk >= 0 && tk < msize) c = ld_idx(&map, tk);
                }
                mixed[k] = wrap_mul(c, ld_idx(&mult, k));
            }
            int64_t mix = mixed[0];
            for (int64_t g = 1; g < ngram; ++g) {
                mix ^= mixed[g];
                for (int64_t j = 0; j < per; ++j) {
                    const int64_t h = (g - 1) * per + j;
                    st_int(ids.dtype, ids.data, m * ids.stride[0] + h * ids.stride[1],
                           floor_mod(mix, ld_idx(&vsz, h)) + ld_idx(&off, h));
                }
            }
            const int64_t t0 = ld_idx(&tok, m);
            st_int(mask.dtype, mask.data, offlin(&mask, m),
                   t0 != pad && t0 != image && t0 != video ? 1 : 0);
        }
        /* The window after the step, by ple_conv's rule: a decode step (num_accepted present)
         * shifts the old ids down by one and appends every id it ran; otherwise every id commits
         * and the last ctx land at offset zero. */
        if (nacc) {
            for (int64_t i = 0; i + 1 < ctx; ++i) st_int(st.dtype, st.data, sb + i, prev[i + 1]);
            for (int64_t t = 0; t < T; ++t)
                st_int(st.dtype, st.data, sb + ctx - 1 + t, ld_idx(&tok, bos + t));
        } else {
            for (int64_t i = 0; i < ctx; ++i) {
                const int64_t src = T - ctx + i;
                st_int(st.dtype, st.data, sb + i, src >= 0 ? ld_idx(&tok, bos + src) : prev[src + ctx]);
            }
        }
    }
    return RAD_OK;
}

/* One rms a head: y[i] = round(gain[i] * x[i] * rsqrt(mean_head(x^2) + eps)). */
int ref_engram_norm(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &e = a->t[0], &gain = a->t[1], &y = a->t[2];
    const int64_t H = p_int(a, "heads", 0), n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 0);
    if (H < 1 || n < 1 || !std::isfinite(eps) || eps < 0 || e.rank != 2 || e.shape[0] < 1)
        return RAD_E_SHAPE;
    const int64_t M = e.shape[0];
    if (!matrix(e, M, H * n) || !matrix(y, M, H * n) || gain.rank != 1 || numel(&gain) != H * n)
        return RAD_E_SHAPE;
    REF_PAR_FOR(M * H * n)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t h = 0; h < H; ++h) {
            double sq = 0;
            for (int64_t i = 0; i < n; ++i) {
                const double v = ldt(&e, m * e.stride[0] + (h * n + i) * e.stride[1]);
                sq += v * v;
            }
            const float inv = 1.0f / std::sqrt((float)(sq / (double)n) + eps);
            for (int64_t i = 0; i < n; ++i)
                stt(&y, m * y.stride[0] + (h * n + i) * y.stride[1],
                    ldt(&gain, (h * n + i) * gain.stride[0]) *
                        ldt(&e, m * e.stride[0] + (h * n + i) * e.stride[1]) * inv);
        }
    return RAD_OK;
}

/* The gate of HYV4FlashEngramLayer.forward and its add into the streams (see the file's OPS.md
 * entry); rows with mask 0 are left untouched. */
int ref_engram_gate(const RadArgs* a, RadStream) {
    if (a->n_t < 8 || !have(a, 8)) return RAD_E_INVAL;
    const RadTensor &key = a->t[0], &val = a->t[1], &mask = a->t[2], &wk = a->t[3], &wq = a->t[4],
                    &wv = a->t[5], &wb = a->t[6], &h = a->t[7];
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 0);
    if (n < 1 || hc < 1 || !std::isfinite(eps) || eps < 0 || h.rank != 2 || h.shape[0] < 1)
        return RAD_E_SHAPE;
    const int64_t M = h.shape[0], W = hc * n;
    if (!matrix(h, M, W) || !matrix(key, M, W) || !matrix(val, M, n) || numel(&mask) < M ||
        numel(&wk) != W || numel(&wq) != W || numel(&wv) != n || numel(&wb) != W)
        return RAD_E_SHAPE;
    const float inv_sqrt_n = (float)std::pow((double)n, -0.5);
    REF_PAR_FOR(M * hc * n * 4)
    for (int64_t m = 0; m < M; ++m) {
        if (!ld_idx(&mask, m)) continue;
        std::vector<float> main((size_t)n);
        {
            double sq = 0;
            for (int64_t i = 0; i < n; ++i) {
                const double v = ldt(&val, m * val.stride[0] + i * val.stride[1]);
                sq += v * v;
            }
            const float inv = 1.0f / std::sqrt((float)(sq / (double)n) + eps);
            for (int64_t i = 0; i < n; ++i)
                main[(size_t)i] = ldt(&wv, i * wv.stride[0]) *
                                  ldt(&val, m * val.stride[0] + i * val.stride[1]) * inv;
        }
        std::vector<float> upd((size_t)W);
        for (int64_t c = 0; c < hc; ++c) {
            double sk = 0, sh = 0;
            for (int64_t i = 0; i < n; ++i) {
                const double kv = ldt(&key, m * key.stride[0] + (c * n + i) * key.stride[1]);
                const double hv = ldt(&h, m * h.stride[0] + (c * n + i) * h.stride[1]);
                sk += kv * kv;
                sh += hv * hv;
            }
            const float ik = 1.0f / std::sqrt((float)(sk / (double)n) + eps);
            const float iq = 1.0f / std::sqrt((float)(sh / (double)n) + eps);
            double dot = 0;
            for (int64_t i = 0; i < n; ++i) {
                const float kn = ldt(&wk, (c * n + i) * wk.stride[0]) *
                                 ldt(&key, m * key.stride[0] + (c * n + i) * key.stride[1]) * ik;
                const float qn = ldt(&wq, (c * n + i) * wq.stride[0]) *
                                 ldt(&h, m * h.stride[0] + (c * n + i) * h.stride[1]) * iq;
                dot += (double)kn * (double)qn;
            }
            const float d = (float)dot * inv_sqrt_n;
            const float g = act_sigmoid(sign_of(d) * std::sqrt(std::max(std::fabs(d), 1e-6f)));
            for (int64_t i = 0; i < n; ++i)
                upd[(size_t)(c * n + i)] =
                    round_dt(h.dtype, g * main[(size_t)i] +
                                           ldt(&wb, (c * n + i) * wb.stride[0]) * main[(size_t)i]);
        }
        for (int64_t i = 0; i < W; ++i) {
            const int64_t o = m * h.stride[0] + i * h.stride[1];
            stt(&h, o, ldt(&h, o) + upd[(size_t)i]);
        }
    }
    return RAD_OK;
}
