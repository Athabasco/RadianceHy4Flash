/* Drives libr4d's absorbed MLA with a sink (r4d_mla_sink.hip: mla_attn_paged_sink and
 * mla_attn_sel_sink) through the SIMT emulator. Every output element against an f64 softmax with
 * the sink in its denominator; the launch forms (16 or 32 heads a workgroup, element-wise key load)
 * against each other bit for bit; the unsplit form within the bound; verify rows against the
 * single-row decode at each position, bit for bit; the paged form at window 0 against the
 * selection form over the same pages, bit for bit; the window edge with boosted keys on both sides
 * of it; sinks far above and below the scores; unmapped and out-of-range pages; heads 8 to 64. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" void r4d_mla_sink_pin_form(int form);

namespace {

constexpr int L = 512, R = 64, W = L + R;
constexpr double kScale = 0.0625;

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}

/* One launch's operands. Paged: sequences with query lengths and contexts, one table each. The
 * selection form reads `sel_bt` / `sel_len`, one table row a query row. */
struct Problem {
    int heads = 8, bs = 4, window = 0, pages = 0, blocks = 0;
    std::vector<int> qlen, ctx;
    std::vector<int> bt, used, cu;          /* paged */
    std::vector<int> sel_bt, sel_len;       /* selection */
    int sel_blocks = 0;
    std::vector<uint16_t> q, qr, kv;
    std::vector<float> sinks;
    int rows() const { return cu.empty() ? 0 : cu.back(); }
};

uint16_t bf(float v) { return rad_f32_to_bf16(v); }
float fb(uint16_t v) { return rad_bf16_to_f32(v); }

/* Pages are a scramble of the sequence's blocks; `holes` unmaps a block and points one past the
 * cache. */
void make_paged(Problem& p, std::vector<int> qlen, std::vector<int> ctx, int heads, int bs,
                int window, bool holes, std::mt19937& rng, float qamp = 0.5f) {
    p.heads = heads;
    p.bs = bs;
    p.window = window;
    p.qlen = qlen;
    p.ctx = ctx;
    const int S = (int)qlen.size();
    p.blocks = 1;
    int total = 0;
    for (int s = 0; s < S; ++s) {
        p.blocks = std::max(p.blocks, (ctx[s] + bs - 1) / bs);
        total += (ctx[s] + bs - 1) / bs;
    }
    p.pages = std::max(total, 1) + 3;
    std::vector<int> order(p.pages);
    for (int i = 0; i < p.pages; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    p.bt.assign((size_t)S * p.blocks, -1);
    int next = 0;
    for (int s = 0; s < S; ++s)
        for (int b = 0; b < (ctx[s] + bs - 1) / bs; ++b) p.bt[(size_t)s * p.blocks + b] = order[next++];
    if (holes)
        for (int s = 0; s < S; ++s) {
            const int nb = (ctx[s] + bs - 1) / bs;
            if (nb > 3) p.bt[(size_t)s * p.blocks + 1] = -1;
            if (nb > 6) p.bt[(size_t)s * p.blocks + nb - 3] = p.pages + 5;
        }
    p.used = ctx;
    p.cu.assign(S + 1, 0);
    for (int s = 0; s < S; ++s) p.cu[s + 1] = p.cu[s] + qlen[s];
    const int M = p.rows();
    std::normal_distribution<float> nrm(0.f, 1.f);
    p.q.resize((size_t)M * heads * L);
    p.qr.resize((size_t)M * heads * R);
    p.kv.resize((size_t)p.pages * bs * W);
    for (auto& v : p.q) v = bf(qamp * nrm(rng));
    for (auto& v : p.qr) v = bf(qamp * nrm(rng));
    for (auto& v : p.kv) v = bf(nrm(rng));
    /* sinks: ordinary, far above (30, 100), far below, and the ordinary scale again */
    const float table[] = { 0.5f, 30.f, -30.f, 2.f, 100.f, -1.f, 4.f, -100.f };
    p.sinks.resize(heads);
    for (int h = 0; h < heads; ++h) p.sinks[h] = table[h % 8] + 0.01f * h;
}

/* Selection tables equal to each row's sequence table, length p + 1: window 0's keys. */
void make_sel_from_paged(Problem& p) {
    const int M = p.rows(), S = (int)p.qlen.size();
    p.sel_blocks = p.blocks;
    p.sel_bt.assign((size_t)M * p.blocks, -1);
    p.sel_len.assign(M, 0);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < p.qlen[s]; ++i) {
            const int r = p.cu[s] + i;
            std::copy(p.bt.begin() + (size_t)s * p.blocks, p.bt.begin() + (size_t)(s + 1) * p.blocks,
                      p.sel_bt.begin() + (size_t)r * p.blocks);
            p.sel_len[r] = p.used[s] - p.qlen[s] + i + 1;
        }
}

/* f64: the keys [lo, hi) of table row `row`, the sink in the maximum and the denominator. */
void ref_one(const Problem& p, const int* table, int lo, int hi, int r, int h, double* out) {
    std::vector<double> s;
    std::vector<const uint16_t*> rows;
    const uint16_t* qv = &p.q[((size_t)r * p.heads + h) * L];
    const uint16_t* rv = &p.qr[((size_t)r * p.heads + h) * R];
    for (int t = lo; t < hi; ++t) {
        const int pg = table[t / p.bs];
        if (pg < 0 || pg >= p.pages) continue;
        const uint16_t* k = &p.kv[((size_t)pg * p.bs + t % p.bs) * W];
        double d = 0;
        for (int c = 0; c < L; ++c) d += (double)fb(qv[c]) * fb(k[c]);
        for (int c = 0; c < R; ++c) d += (double)fb(rv[c]) * fb(k[L + c]);
        s.push_back(d * kScale);
        rows.push_back(k);
    }
    for (int c = 0; c < L; ++c) out[c] = 0;
    if (s.empty()) return;
    const double sink = p.sinks[h];
    double mx = sink;
    for (double v : s) mx = std::max(mx, v);
    double den = std::exp(sink - mx);
    for (size_t j = 0; j < s.size(); ++j) {
        const double w = std::exp(s[j] - mx);
        den += w;
        for (int c = 0; c < L; ++c) out[c] += w * fb(rows[j][c]);
    }
    for (int c = 0; c < L; ++c) out[c] /= den;
}

std::vector<double> ref_paged(const Problem& p) {
    const int M = p.rows(), S = (int)p.qlen.size();
    std::vector<double> want((size_t)M * p.heads * L);
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < p.qlen[s]; ++i) {
            const int r = p.cu[s] + i;
            const long pos = (long)p.used[s] - p.qlen[s] + i;
            const int hi = (int)std::min<long>(std::max<long>(pos + 1, 0), (long)p.blocks * p.bs);
            const int lo = p.window > 0 ? (int)std::max<long>(0, pos + 1 - p.window) : 0;
            for (int h = 0; h < p.heads; ++h)
                ref_one(p, &p.bt[(size_t)s * p.blocks], lo, hi, r, h,
                        &want[((size_t)r * p.heads + h) * L]);
        }
    return want;
}

std::vector<double> ref_sel(const Problem& p) {
    const int M = p.rows();
    std::vector<double> want((size_t)M * p.heads * L);
    for (int r = 0; r < M; ++r) {
        const int hi = std::clamp(p.sel_len[r], 0, p.sel_blocks * p.bs);
        for (int h = 0; h < p.heads; ++h)
            ref_one(p, &p.sel_bt[(size_t)r * p.sel_blocks], 0, hi, r, h,
                    &want[((size_t)r * p.heads + h) * L]);
    }
    return want;
}

struct Params {
    std::vector<RadParam> v;
    void i(const char* k, long long x) {
        RadParam p{};
        p.key = k;
        p.kind = RAD_P_INT;
        p.ival = x;
        v.push_back(p);
    }
    void f(const char* k, double x) {
        RadParam p{};
        p.key = k;
        p.kind = RAD_P_F64;
        p.dval = x;
        v.push_back(p);
    }
};

int launch(const Problem& p, bool sel, int form, std::vector<uint16_t>& out) {
    const int M = p.rows();
    out.assign((size_t)M * p.heads * L, 0xffff);
    auto& P = const_cast<Problem&>(p);
    std::vector<RadTensor> t;
    t.push_back(tens(P.q.data(), RAD_BF16, { M, p.heads, L }));
    t.push_back(tens(P.qr.data(), RAD_BF16, { M, p.heads, R }));
    t.push_back(tens(P.kv.data(), RAD_BF16, { p.pages, 1, p.bs, W }));
    Params ps;
    ps.i("M", M);
    ps.i("heads", p.heads);
    ps.i("latent_dim", L);
    ps.i("rope_dim", R);
    if (sel) {
        t.push_back(tens(P.sel_bt.data(), RAD_I32, { M, p.sel_blocks }));
        t.push_back(tens(P.sel_len.data(), RAD_I32, { M }));
        ps.i("blocks", p.sel_blocks);
    } else {
        t.push_back(tens(P.bt.data(), RAD_I32, { (int64_t)p.qlen.size(), p.blocks }));
        t.push_back(tens(P.used.data(), RAD_I32, { (int64_t)p.used.size() }));
        t.push_back(tens(P.cu.data(), RAD_I32, { (int64_t)p.cu.size() }));
        ps.i("window", p.window);
    }
    ps.i("block_size", p.bs);
    ps.f("scale", kScale);
    t.push_back(tens(P.sinks.data(), RAD_F32, { p.heads }));
    t.push_back(tens(out.data(), RAD_BF16, { M, p.heads, L }));
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = ps.v.data();
    a.n_p = (int)ps.v.size();
    const int64_t need =
        sel ? r4d_rad_mla_attn_sel_sink_scratch(&a) : r4d_rad_mla_attn_paged_sink_scratch(&a);
    std::vector<char> scratch((size_t)need + 16);
    a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
    a.scratch_bytes = need;
    r4d_mla_sink_pin_form(form);
    const int rc = sel ? r4d_rad_mla_attn_sel_sink(&a, nullptr)
                       : r4d_rad_mla_attn_paged_sink(&a, nullptr);
    r4d_mla_sink_pin_form(0);
    return rc;
}

int g_fail = 0;
void report(const std::string& name, bool ok, const char* detail = "") {
    std::printf("%-64s %s %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail);
    g_fail += !ok;
}

bool close_to(const std::vector<uint16_t>& got, const std::vector<double>& want, double& max_err) {
    long bad = 0;
    max_err = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = fb(got[i]);
        const double e = std::fabs(g - want[i]);
        if (!std::isfinite(g) || e > 1e-5 + 0.008 * std::fabs(want[i])) ++bad;
        if (std::isfinite(g)) max_err = std::max(max_err, e);
    }
    return bad == 0;
}

/* Every form against the reference; forms 0..3 (same split count) bit for bit. */
void check_forms(const std::string& name, const Problem& p, bool sel) {
    const std::vector<double> want = sel ? ref_sel(p) : ref_paged(p);
    std::vector<uint16_t> first;
    for (int form : { 0, 1, 2, 3, 4 }) {
        if (form == 2 && p.heads <= 16) continue;
        std::vector<uint16_t> out;
        const int rc = launch(p, sel, form, out);
        char buf[128];
        if (rc != RAD_OK) {
            std::snprintf(buf, sizeof buf, "rc %d", rc);
            report(name + " form " + std::to_string(form), false, buf);
            continue;
        }
        double err;
        bool ok = close_to(out, want, err);
        std::snprintf(buf, sizeof buf, "max_abs %.2e", err);
        std::string note = buf;
        if (form == 0) first = out;
        else if (form != 4 && out != first) {
            ok = false;
            note += " (bytes differ from the rule's form)";
        }
        report(name + (sel ? " sel" : " paged") + " form " + std::to_string(form), ok, note.c_str());
    }
}

/* Each verify row's bytes against a one-row launch at its position over the same table. */
void check_verify(const std::string& name, const Problem& p) {
    std::vector<uint16_t> all;
    if (launch(p, false, 0, all) != RAD_OK) {
        report(name + " verify", false, "launch");
        return;
    }
    bool same = true;
    const int S = (int)p.qlen.size();
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < p.qlen[s]; ++i) {
            Problem one = p;
            one.qlen = { 1 };
            one.ctx = { p.used[s] - p.qlen[s] + i + 1 };
            one.used = one.ctx;
            one.cu = { 0, 1 };
            one.bt.assign(p.bt.begin() + (size_t)s * p.blocks,
                          p.bt.begin() + (size_t)(s + 1) * p.blocks);
            const int r = p.cu[s] + i;
            one.q.assign(p.q.begin() + (size_t)r * p.heads * L,
                         p.q.begin() + (size_t)(r + 1) * p.heads * L);
            one.qr.assign(p.qr.begin() + (size_t)r * p.heads * R,
                          p.qr.begin() + (size_t)(r + 1) * p.heads * R);
            std::vector<uint16_t> o;
            if (launch(one, false, 0, o) != RAD_OK ||
                !std::equal(o.begin(), o.end(), all.begin() + (size_t)r * p.heads * L))
                same = false;
        }
    report(name + " verify rows == decode rows (bytes)", same);
}

void check_paged_vs_sel(const std::string& name, Problem p) {
    make_sel_from_paged(p);
    std::vector<uint16_t> a, b;
    const int ra = launch(p, false, 0, a), rb = launch(p, true, 0, b);
    report(name + " paged(window 0) == sel (bytes)", ra == RAD_OK && rb == RAD_OK && a == b);
    check_forms(name, p, true);
}

/* The window edge: row at position pos; the key at pos-window (outside) and pos-window+1 (inside)
 * get rotary parts aligned with every head's rotary query, the outside one more strongly, so
 * either mistake dominates the output. */
void check_edge(int heads, int window, int ctx, std::mt19937& rng) {
    Problem p;
    make_paged(p, { 1 }, { ctx }, heads, 4, window, false, rng);
    for (auto& v : p.qr) v = bf(0.5f);
    const int pos = ctx - 1;
    auto boost = [&](int t, float x) {
        const int pg = p.bt[t / p.bs];
        uint16_t* k = &p.kv[((size_t)pg * p.bs + t % p.bs) * W];
        for (int c = 0; c < R; ++c) k[L + c] = bf(x);
    };
    boost(pos - window, 5.f);     /* rotary score 0.5*5*64*scale = 10 */
    boost(pos - window + 1, 3.f); /* 6 */
    for (auto& s : p.sinks) s = 0.f;
    const std::vector<double> want = ref_paged(p);
    std::vector<uint16_t> out;
    const int rc = launch(p, false, 0, out);
    double err;
    const bool ok = rc == RAD_OK && close_to(out, want, err);
    char buf[64];
    std::snprintf(buf, sizeof buf, "max_abs %.2e", rc == RAD_OK ? err : -1.0);
    report("window edge H" + std::to_string(heads) + " ctx " + std::to_string(ctx), ok, buf);
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261121);
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    /* heads x (sequences: query lengths, contexts) x window */
    struct C {
        int heads;
        std::vector<int> qlen, ctx;
        int window, bs;
        bool holes;
    };
    std::vector<C> cases = {
        { 8, { 1 }, { 1 }, 512, 4, false },
        { 8, { 1, 1, 1 }, { 15, 16, 17 }, 512, 4, true },
        { 16, { 2, 1 }, { 511, 513 }, 512, 4, true },
        { 32, { 4 }, { 600 }, 512, 16, true },
        { 64, { 1, 3 }, { 512, 520 }, 512, 4, true },
        { 64, { 2 }, { 300 }, 0, 4, true },
        { 32, { 1, 2 }, { 33, 530 }, 0, 4, true },
    };
    if (quick) cases.resize(3);
    for (const auto& c : cases) {
        Problem p;
        make_paged(p, c.qlen, c.ctx, c.heads, c.bs, c.window, c.holes, rng);
        std::string name = "H" + std::to_string(c.heads) + " w" + std::to_string(c.window) +
                           " bs" + std::to_string(c.bs) + " ctx";
        for (int x : c.ctx) name += " " + std::to_string(x);
        check_forms(name, p, false);
        check_verify(name, p);
        if (c.window == 0) check_paged_vs_sel(name, p);
    }
    /* A verify of 8 rows past the window, and the same rows at window 0 against the selection. */
    {
        Problem p;
        make_paged(p, { 8 }, { 530 }, 16, 4, 512, true, rng);
        check_verify("H16 w512 verify 8 at 530", p);
        p.window = 0;
        check_verify("H16 w0 verify 8 at 530", p);
        check_paged_vs_sel("H16 w0 verify 8 at 530", p);
    }
    check_edge(16, 512, 600, rng);
    if (!quick) check_edge(64, 512, 530, rng);
    /* The selection form on its own tables: scattered pages, holes, a zero length. */
    {
        Problem p;
        make_paged(p, { 3 }, { 200 }, 32, 4, 0, false, rng);
        make_sel_from_paged(p);
        for (int r = 0; r < 3; ++r)
            for (int b = 0; b < p.sel_blocks; ++b)
                p.sel_bt[(size_t)r * p.sel_blocks + b] =
                    b % 13 == 5 ? -1 : b % 11 == 7 ? p.pages + 1 : (b * 7 + r * 3) % p.pages;
        p.sel_len = { 0, 77, 200 };
        check_forms("sel scattered", p, true);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
