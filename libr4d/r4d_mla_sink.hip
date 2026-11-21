/* Absorbed MLA with a per-head sink logit (Hy4-flash): mla_attn_paged_sink (a paged latent cache,
 * one table a sequence, a sliding window counting the row's own token or none) and
 * mla_attn_sel_sink (a per-row table and length: the indexer's selection). Cache row of a token:
 * [latent 512 | rotated key 64] bf16. Design: kernels/mla_sink_attn.md of the Hy4-flash plan.
 *
 * The kernel is attend_rows (r4d_mla.hip) with three changes: the rotary columns join the score
 * (one more WMMA a wave a chunk), the token range has a window lower bound, and the sink joins the
 * softmax once, at the store of an unsplit launch or in the merge of a split one. The sink-in-reduce
 * rule follows vLLM's rocm_aiter_mla_sparse.py (@820fce0d, Apache-2.0): the splits' partials are
 * relative to their own maxima, the merge reads all maxima first, and the sink is applied after the
 * merge, never per split. The code is new.
 *
 * Workgroup: one query row and HG groups of sixteen heads, four waves a group. Scores are
 * S^T = K Q^T: tokens are the WMMA rows, heads the columns, so a head's running maximum and
 * denominator live in its lane. Wave `quarter` of a group takes latent columns
 * [128 quarter, 128 quarter + 128) and rotary columns [16 quarter, 16 quarter + 16); the four
 * partial dots meet in LDS and every wave adds them in the order 0, 1, 2, 3, so all four hold the
 * same scores. P goes to the value product as bf16 high part plus bf16 remainder. The next chunk's
 * keys are fetched into registers while the current one is multiplied; two barriers a chunk.
 *
 * A row's bytes do not depend on the launch: the split count is a function of the op and the
 * window alone (kSplitsWindow, kSplitsFull) and HG changes only which heads share a workgroup. So
 * a verify of k + 1 rows gives the single-row decode at each position bit for bit, and with the
 * same split count the paged form at window 0 and the selection form over a table listing the
 * same pages in order give the same bytes. Launches of more rows than kRowsWindow / kRowsFull
 * (several streams' verify, prefill through the absorbed path) and launches whose scratch is short
 * run unsplit, so their rows' bytes differ from a split launch's in the last place. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_gdn_wmma.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

namespace {
/* A partial: 512 values, max, den, padded to 16 bytes for the merge's vector loads. */
constexpr int D = 512, RD = 64, W = D + RD, PART = D + 4;
/* Row pitch of a staged token in LDS: 576 + 8, so consecutive rows do not share a bank pattern. */
constexpr int kPitch = W + 8;
constexpr int kPieces = 16 * W / 8; /* 16-byte pieces of a 16-token chunk */
/* Splits: a window's 512 keys in 64-token pieces, a selection's 2048 in 128-token ones. A lone
 * workgroup takes about 1.8 us a 16-token chunk (gfx1201), so a decode row's time is the chunks of
 * one split; more splits cost the partials' write and merge. */
constexpr int kSplitsWindow = 8, kSplitsFull = 16;
static_assert(kSplitsWindow <= 16 && kSplitsFull <= 16, "merge_sink holds 16 partials");
/* The most rows a launch splits (a verify at depth 15 or 31, or that many decode rows); past them
 * the workgroups fill the card unsplit, which is then also faster (measured: the window at 16 rows
 * 79 us split against 68 unsplit, the selection at 64 rows 362 against 269). */
constexpr long kRowsWindow = 16, kRowsFull = 32;
/* Table entries a split reads through LDS: a selection split of 128 tokens at block size 1. */
constexpr int kSpanPages = 160;

struct SinkMeta {
    long qr, qh, rr, rh, kp, kt, bp, bc, lp, orr, oh, ss;
    int rows, heads, seqs, blocks, bs, bs_shift, pages, splits, window;
    float scale;
};

/* r4d_mla_sink_pin_form: 0 the launcher's rule, 1 sixteen heads a workgroup, 2 thirty-two, 3 the
 * element-wise key load, 4 unsplit, 5 sixty-four heads a workgroup. A test hook (r4d_selftest, r4d_emu); nothing else sets it. */
int g_sink_form = 0;

__device__ __forceinline__ int sink_seq(const int* cu, int seqs, int r) {
    int s = 0;
    while (s + 1 < seqs && cu[s + 1] <= r) ++s;
    return s;
}

/* The visible token range of row r and the table row it reads. */
template <bool SEL>
__device__ __forceinline__ void sink_range(const int* used, const int* cu, const int* len,
                                           const SinkMeta& m, int r, int& lo, int& hi, int& trow) {
    const int cap = m.blocks * m.bs;
    if constexpr (SEL) {
        lo = 0;
        hi = min(max(len[(long)r * m.lp], 0), cap);
        trow = r;
    } else {
        const int s = sink_seq(cu, m.seqs, r);
        const int qlen = cu[s + 1] - cu[s], i = r - cu[s];
        const long p = (long)used[s] - qlen + i;
        const long top = p + 1 < 0 ? 0 : p + 1 < cap ? p + 1 : cap,
                   bottom = m.window > 0 && p + 1 - m.window > 0 ? p + 1 - m.window : 0;
        hi = (int)top;
        lo = (int)(bottom < top ? bottom : top);
        trow = s;
    }
}

template <int HG, bool WIDE, bool SEL>
__global__ __launch_bounds__(HG * 128) void mla_sink_rows(
    const uint16_t* q, const uint16_t* qrot, const uint16_t* kv, const int* bt, const int* used,
    const int* cu, const int* len, const float* sinks, uint16_t* out, float* scratch, SinkMeta m) {
    constexpr int NT = HG * 128, LOADS = (kPieces + NT - 1) / NT;
    static_assert(LOADS <= 32, "staged mask bits");
    __shared__ __align__(16) uint16_t keys[2][16 * kPitch];
    __shared__ __align__(32) float partial[HG * 4][32 * 8];
    __shared__ int visible[2][16];
    __shared__ long rows_at[2][16];
    __shared__ int span_pages[kSpanPages];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, quarter = wave & 3,
              group = wave >> 2, r = blockIdx.x, split = blockIdx.z;
    const int head = blockIdx.y * 16 * HG + group * 16 + (lane & 15);
    const bool mine = head < m.heads;
    int lo, hi, trow;
    sink_range<SEL>(used, cu, len, m, r, lo, hi, trow);
    const long n = hi - lo;
    const int begin = lo + (int)(n * split / m.splits), end = lo + (int)(n * (split + 1) / m.splits);
    /* Query fragments: eight latent k-steps of the wave's quarter, one rotary. */
    v8bf query[9];
    {
        const uint16_t* qp =
            q + (long)r * m.qr + (long)(mine ? head : 0) * m.qh + quarter * 128 + 4 * (lane >> 4);
        const uint16_t* rp =
            qrot + (long)r * m.rr + (long)(mine ? head : 0) * m.rh + quarter * 16 + 4 * (lane >> 4);
#pragma unroll
        for (int s = 0; s < 9; ++s) {
            const uint16_t* src = s < 8 ? qp + s * 16 : rp;
            union {
                uint16_t h[8];
                v8bf f;
            } u;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                u.h[e] = mine ? src[e] : (uint16_t)0;
                u.h[4 + e] = mine ? src[8 + e] : (uint16_t)0;
            }
            query[s] = u.f;
        }
    }
    const int* const table = bt + (long)trow * m.bp;
    /* The cache offset of each token of a chunk, or -1 where nothing is visible, read from the
     * table two chunks ahead by sixteen lanes and passed through LDS, so a key load never waits
     * on a table load. */
    /* A split's whole table slice, where it fits, is read into LDS at once in the prologue: one
     * table round trip a workgroup instead of one every few chunks (a table read cold from
     * memory otherwise stalls the chunk that crosses into its next cache line). */
    const int first_block = begin >> m.bs_shift,
              span = begin < end ? ((end - 1) >> m.bs_shift) - first_block + 1 : 0;
    const bool spanned = span <= kSpanPages;
    auto token_at = [&](int token) -> long {
        if (token >= end) return -1;
        const int block = token >> m.bs_shift;
        const int page = spanned ? span_pages[block - first_block] : table[(long)block * m.bc];
        return page >= 0 && page < m.pages
                   ? (long)page * m.kp + (long)(token & (m.bs - 1)) * m.kt
                   : -1;
    };
    uint4 staged[LOADS];
    unsigned staged_ok = 0;
    auto fetch = [&](int slot) {
#pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int flat = tid + j * NT;
            if (LOADS * NT > kPieces && flat >= kPieces) continue;
            const int piece = flat % (W / 8);
            const long row = rows_at[slot][flat / (W / 8)];
            const bool ok = row >= 0;
            const long at = ok ? row + piece * 8 : 0;
            if constexpr (WIDE) {
                staged[j] = *(const uint4*)(kv + at);
                if (!ok) staged[j] = make_uint4(0, 0, 0, 0);
            } else {
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    w[i] = ok ? (uint32_t)kv[at + 2 * i] | (uint32_t)kv[at + 2 * i + 1] << 16 : 0u;
                staged[j] = make_uint4(w[0], w[1], w[2], w[3]);
            }
            staged_ok = ok ? staged_ok | 1u << j : staged_ok & ~(1u << j);
        }
    };
    auto stash = [&](int b) {
#pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int flat = tid + j * NT;
            if (LOADS * NT > kPieces && flat >= kPieces) continue;
            const int token = flat / (W / 8), piece = flat % (W / 8);
            *(uint4*)(keys[b] + token * kPitch + piece * 8) = staged[j];
            if (!piece) visible[b][token] = (int)(staged_ok >> j & 1u);
        }
    };
    v8f value[8];
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) value[mt] = (v8f){ 0 };
    float run_max = -INFINITY, run_den = 0;
    if (begin < end) {
        if (spanned)
            for (int i = tid; i < span; i += NT) span_pages[i] = table[(long)(first_block + i) * m.bc];
        __syncthreads();
        if (tid < 32) rows_at[tid >> 4][tid & 15] = token_at(begin + tid);
        __syncthreads();
        fetch(0);
        stash(0);
    }
    __syncthreads();
    int buf = 0;
    for (int start = begin; start < end; start += 16) {
        const bool more = start + 16 < end;
        if (more) fetch(buf ^ 1);
        long ahead = -1;
        if (tid < 16 && start + 32 < end) ahead = token_at(start + 32 + tid);
        const uint16_t* tile = keys[buf];
        v8f dot = { 0 };
#pragma unroll
        for (int s = 0; s < 8; ++s)
            dot = mma(frag(tile, kPitch, 0, quarter * 128 + s * 16, lane), query[s], dot);
        dot = mma(frag(tile, kPitch, 0, D + quarter * 16, lane), query[8], dot);
        *(v8f*)&partial[wave][lane * 8] = dot;
        __syncthreads();
        v8f score = *(const v8f*)&partial[group * 4][lane * 8];
#pragma unroll
        for (int k = 1; k < 4; ++k) score += *(const v8f*)&partial[group * 4 + k][lane * 8];
        float chunk_max = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            score[e] = visible[buf][8 * (lane >> 4) + e] ? score[e] * m.scale : -INFINITY;
            chunk_max = fmaxf(chunk_max, score[e]);
        }
        chunk_max = fmaxf(chunk_max, swap16(chunk_max));
        const float next_max = fmaxf(run_max, chunk_max);
        const float alpha = isfinite(next_max) ? expf(run_max - next_max) : 1;
        float p[8], sum = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p[e] = isfinite(score[e]) ? expf(score[e] - next_max) : 0;
            sum += p[e];
        }
        /* Both halves add tokens 0-7 before 8-15, so the sixteen lanes of a head agree. */
        const float other = swap16(sum);
        run_den = run_den * alpha + (lane < 16 ? sum + other : other + sum);
        run_max = next_max;
        /* A B fragment's lane holds its head's tokens 0-3 and 8-11 (lanes 0-15) or 4-7 and 12-15
         * (16-31); a score lane holds 0-7 or 8-15. Each half sends the other the four it needs. */
        uint32_t phi[4], plo[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint16_t h0 = f32_to_bf16(p[2 * i]), h1 = f32_to_bf16(p[2 * i + 1]);
            phi[i] = (uint32_t)h0 | (uint32_t)h1 << 16;
            plo[i] = (uint32_t)f32_to_bf16(p[2 * i] - bf16_to_f32(h0)) |
                     (uint32_t)f32_to_bf16(p[2 * i + 1] - bf16_to_f32(h1)) << 16;
        }
        const bool low = lane < 16;
        const uint32_t hx0 = r4d_xor_lane_u32<16>(low ? phi[2] : phi[0]),
                       hx1 = r4d_xor_lane_u32<16>(low ? phi[3] : phi[1]),
                       lx0 = r4d_xor_lane_u32<16>(low ? plo[2] : plo[0]),
                       lx1 = r4d_xor_lane_u32<16>(low ? plo[3] : plo[1]);
        union {
            uint32_t w[4];
            v8bf f;
        } ph, pl;
        ph.w[0] = low ? phi[0] : hx0;
        ph.w[1] = low ? phi[1] : hx1;
        ph.w[2] = low ? hx0 : phi[2];
        ph.w[3] = low ? hx1 : phi[3];
        pl.w[0] = low ? plo[0] : lx0;
        pl.w[1] = low ? plo[1] : lx1;
        pl.w[2] = low ? lx0 : plo[2];
        pl.w[3] = low ? lx1 : plo[3];
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) {
            const auto v = fragT(tile, kPitch, quarter * 128 + mt * 16, 0, lane);
            value[mt] *= alpha;
            value[mt] = mma(v, ph.f, value[mt]);
            value[mt] = mma(v, pl.f, value[mt]);
        }
        if (more) stash(buf ^ 1);
        if (tid < 16) rows_at[buf][tid] = ahead;
        __syncthreads();
        buf ^= 1;
    }
    if (!mine) return;
    if (m.splits == 1) {
        /* The sink joins the softmax once: shift to max(run_max, sink) as libref does, so a sink far
         * above every score cannot overflow. No visible key gives zero, sink or not. */
        float inv = 0;
        if (run_den > 0) {
            const float sink = sinks[(long)head * m.ss], top = fmaxf(run_max, sink),
                        c = expf(run_max - top);
            inv = c / (run_den * c + expf(sink - top));
        }
#pragma unroll
        for (int mt = 0; mt < 8; ++mt)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = quarter * 128 + mt * 16 + 8 * (lane >> 4) + e;
                out[(long)r * m.orr + (long)head * m.oh + col] = f32_to_bf16(value[mt][e] * inv);
            }
        return;
    }
    float* dst = scratch + (((long)r * m.heads + head) * m.splits + split) * PART;
#pragma unroll
    for (int mt = 0; mt < 8; ++mt)
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[quarter * 128 + mt * 16 + 8 * (lane >> 4) + e] = value[mt][e];
    if (!quarter && lane < 16) {
        dst[D] = run_max;
        dst[D + 1] = run_den;
    }
}

/* A workgroup a (row, head), thread t holding columns 4t..4t+3: every split's maximum and partial
 * loaded at once (no serial dependency between splits), the partials rescaled to the global
 * maximum and added in split order, then the sink once. */
constexpr int kMaxSplits = 16;
__global__ __launch_bounds__(128) void merge_sink(const float* scratch, const float* sinks,
                                                  uint16_t* out, SinkMeta m) {
    const int t = threadIdx.x, head = blockIdx.y, r = blockIdx.x;
    const float* src = scratch + ((long)r * m.heads + head) * m.splits * PART;
    float part_max[kMaxSplits], part_den[kMaxSplits];
    float4 part[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
        if (s < m.splits) {
            part_max[s] = src[s * PART + D];
            part_den[s] = src[s * PART + D + 1];
            part[s] = *(const float4*)(src + s * PART + 4 * t);
        }
    float top = -INFINITY;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
        if (s < m.splits && part_den[s] > 0) top = fmaxf(top, part_max[s]);
    float4 value = make_float4(0, 0, 0, 0);
    float den = 0;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) {
        if (!(s < m.splits && part_den[s] > 0)) continue;
        const float w = expf(part_max[s] - top);
        den += part_den[s] * w;
        value.x += part[s].x * w;
        value.y += part[s].y * w;
        value.z += part[s].z * w;
        value.w += part[s].w * w;
    }
    float inv = 0;
    if (den > 0) {
        const float sink = sinks[(long)head * m.ss], shifted = fmaxf(top, sink),
                    c = expf(top - shifted);
        inv = c / (den * c + expf(sink - shifted));
    }
    uint16_t* const o = out + (long)r * m.orr + (long)head * m.oh + 4 * t;
    o[0] = f32_to_bf16(value.x * inv);
    o[1] = f32_to_bf16(value.y * inv);
    o[2] = f32_to_bf16(value.z * inv);
    o[3] = f32_to_bf16(value.w * inv);
}

int sink_splits(bool sel, long window) { return !sel && window > 0 ? kSplitsWindow : kSplitsFull; }
long sink_rows_cap(bool sel, long window) { return !sel && window > 0 ? kRowsWindow : kRowsFull; }

void heads_view(RadTensor& t, long heads, long width) {
    if (heads > 0 && t.rank == 2 && t.shape[1] == heads * width) {
        t.rank = 3;
        t.shape[1] = heads;
        t.shape[2] = width;
        t.stride[2] = t.stride[1];
        t.stride[1] *= width;
    }
}

bool row_tensor_ok(const RadTensor& t, long rows, long heads, long width) {
    return t.rank == 3 && t.shape[0] == rows && t.shape[1] == heads && t.shape[2] == width &&
           t.stride[2] == 1 && t.stride[1] >= width && t.stride[0] >= heads * t.stride[1];
}

/* Operands, paged: q [M, heads, 512], q_rope [M, heads, 64], kv [pages, 1, bs, >= 576],
 * block_table [seqs, blocks], seqused [seqs], cu_seqlens [seqs + 1], sinks [>= heads] f32, out
 * [M, heads, 512]. Selection: q, q_rope, kv, bt [>= M, blocks], length [>= M], sinks, out. */
template <bool SEL>
int sink_body(const RadArgs* a, RadStream stream) {
    const int nt = SEL ? 7 : 8;
    if (!a || a->n_t < nt) return RAD_E_INVAL;
    for (int i = 0; i < nt; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[nt - 1];
    const RadTensor &kv = a->t[2], &bt = a->t[3], &sk = a->t[nt - 2];
    const RadTensor* su = SEL ? nullptr : &a->t[4];
    const RadTensor* cu = SEL ? nullptr : &a->t[5];
    const RadTensor* len = SEL ? &a->t[4] : nullptr;
    const long heads = r4d_geti_or(a, "heads", 0), bs = r4d_geti_or(a, "block_size", 0),
               window = SEL ? 0 : r4d_geti_or(a, "window", 0);
    const float scale = (float)r4d_getf(a, "scale", 0);
    heads_view(q, heads, D);
    heads_view(qr, heads, RD);
    heads_view(o, heads, D);
    if (q.dtype != RAD_BF16 || qr.dtype != RAD_BF16 || kv.dtype != RAD_BF16 ||
        o.dtype != RAD_BF16 || bt.dtype != RAD_I32 || sk.dtype != RAD_F32 ||
        (su && su->dtype != RAD_I32) || (cu && cu->dtype != RAD_I32) ||
        (len && len->dtype != RAD_I32))
        return RAD_E_DTYPE;
    const long rows = q.shape[0];
    if (r4d_geti_or(a, "latent_dim", 0) != D || r4d_geti_or(a, "rope_dim", 0) != RD ||
        heads < 8 || heads > 64 || heads % 8 || bs < 1 || bs > 256 || (bs & (bs - 1)) || window < 0 ||
        window > INT_MAX || !std::isfinite(scale) || scale <= 0 || q.rank != 3 || rows < 1 ||
        rows > 65535 || !row_tensor_ok(q, rows, heads, D) || !row_tensor_ok(qr, rows, heads, RD) ||
        !row_tensor_ok(o, rows, heads, D) || kv.rank != 4 || kv.shape[0] < 1 ||
        kv.shape[0] > INT_MAX || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < W ||
        kv.stride[3] != 1 || kv.stride[2] < W || kv.stride[0] < bs * kv.stride[2] ||
        bt.rank != 2 || bt.shape[1] < 1 || bt.shape[1] > INT_MAX / bs || bt.stride[1] < 1 ||
        bt.stride[0] < bt.shape[1] * bt.stride[1] || sk.rank != 1 || sk.shape[0] < heads ||
        sk.stride[0] < 1)
        return RAD_E_SHAPE;
    if (SEL && (bt.shape[0] < rows || len->rank != 1 || len->shape[0] < rows ||
                len->stride[0] < 1))
        return RAD_E_SHAPE;
    if (!SEL && (su->rank != 1 || su->shape[0] < 1 || bt.shape[0] < su->shape[0] ||
                 !r4d_contig(su) || !r4d_contig(cu) || r4d_numel(cu) != su->shape[0] + 1))
        return RAD_E_SHAPE;
    const int form = g_sink_form;
    int splits = sink_splits(SEL, window);
    if (form == 4 || rows > sink_rows_cap(SEL, window) || !a->scratch ||
        a->scratch_bytes < rows * heads * splits * PART * 4)
        splits = 1;
    const bool wide = form != 3 && !((uintptr_t)kv.data & 15) && !(kv.stride[0] % 8) &&
                      !(kv.stride[2] % 8);
    /* More heads a workgroup load each key chunk once for more of them: the widest group (64,
     * 32 or 16 heads) that still leaves 64 workgroups, two to each of the card's 32 WGPs. The
     * choice moves no bits. */
    int hg = 1;
    for (int g : { 4, 2 })
        if (heads > 16 * (g / 2) && rows * splits * ((heads + 16 * g - 1) / (16 * g)) >= 64) {
            hg = g;
            break;
        }
    if (form == 1) hg = 1;
    if (form == 2 && heads > 16) hg = 2;
    if (form == 5 && heads > 32) hg = 4;
    const SinkMeta m{ q.stride[0],
                      q.stride[1],
                      qr.stride[0],
                      qr.stride[1],
                      kv.stride[0],
                      kv.stride[2],
                      bt.stride[0],
                      bt.stride[1],
                      len ? (long)len->stride[0] : 0L,
                      o.stride[0],
                      o.stride[1],
                      sk.stride[0],
                      (int)rows,
                      (int)heads,
                      su ? (int)su->shape[0] : 0,
                      (int)bt.shape[1],
                      (int)bs,
                      __builtin_ctzl((unsigned long)bs),
                      (int)kv.shape[0],
                      splits,
                      (int)window,
                      scale };
    const auto* qp = (const uint16_t*)q.data;
    const auto* rp = (const uint16_t*)qr.data;
    const auto* kp = (const uint16_t*)kv.data;
    const auto* bp = (const int*)bt.data;
    const int* up = su ? (const int*)su->data : nullptr;
    const int* cp = cu ? (const int*)cu->data : nullptr;
    const int* lp = len ? (const int*)len->data : nullptr;
    const auto* sp = (const float*)sk.data;
    auto* op = (uint16_t*)o.data;
    auto* scr = (float*)a->scratch;
#define SINK_EMIT(HG, WD)                                                                          \
    R4D_EMIT(stream, (mla_sink_rows<HG, WD, SEL>),                                                 \
             dim3((unsigned)rows, (unsigned)((heads + 16 * HG - 1) / (16 * HG)), (unsigned)splits), \
             dim3(HG * 128), 0, qp, rp, kp, bp, up, cp, lp, sp, op, scr, m)
    int rc;
    if (hg == 4)
        rc = wide ? SINK_EMIT(4, true) : SINK_EMIT(4, false);
    else if (hg == 2)
        rc = wide ? SINK_EMIT(2, true) : SINK_EMIT(2, false);
    else
        rc = wide ? SINK_EMIT(1, true) : SINK_EMIT(1, false);
#undef SINK_EMIT
    if (rc == RAD_OK && splits > 1)
        rc = R4D_EMIT(stream, merge_sink, dim3((unsigned)rows, (unsigned)heads), dim3(128), 0,
                      (const float*)scr, sp, op, m);
    return rc;
}

int64_t sink_scratch(const RadArgs* a, bool sel) {
    const long heads = r4d_geti_or(a, "heads", 0), rows = r4d_geti_or(a, "M", kRowsFull),
               window = sel ? 0 : r4d_geti_or(a, "window", 0);
    if (heads < 1 || heads > 64 || rows < 1 || window < 0) return RAD_E_SHAPE;
    return std::min(rows, sink_rows_cap(sel, window)) * heads * sink_splits(sel, window) * PART * 4;
}
}  // namespace

extern "C" int r4d_rad_mla_attn_paged_sink(const RadArgs* a, RadStream stream) {
    return sink_body<false>(a, stream);
}
extern "C" int r4d_rad_mla_attn_sel_sink(const RadArgs* a, RadStream stream) {
    return sink_body<true>(a, stream);
}
/* Sized for min(M, the split row cap) rows at the op's split count: with a window (16 rows, 8
 * splits) 16.1 MiB at 64 heads; at window 0 or a selection (32 rows, 16 splits) 32.3 MiB at 32
 * heads, 64.5 MiB at 64. */
extern "C" int64_t r4d_rad_mla_attn_paged_sink_scratch(const RadArgs* a) {
    return sink_scratch(a, false);
}
extern "C" int64_t r4d_rad_mla_attn_sel_sink_scratch(const RadArgs* a) {
    return sink_scratch(a, true);
}
extern "C" void r4d_mla_sink_pin_form(int form) { g_sink_form = form; }
