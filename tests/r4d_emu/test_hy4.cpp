/* tests/r4d_emu: Hy4-flash's small ops (r4d_ihc_bf16.hip, r4d_engram_bf16.hip, r4d_mla_kv_bf16.hip)
 * on the host SIMT emulator, each held to libref's op (libref/ref_hy4.cpp) on the same bytes.
 * ihc_post, mla_kv_split and Engram's ids, mask and window compare byte for byte; the others within
 * the bound their kernels state (one bf16 ulp of a differently ordered or fused sum). Row padding of
 * a pitched operand must come back untouched. Needs libref (RADIANCE_HOME), as test_kda. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <vector>

static const RadKernelInfo* libref_op(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
        if (open && open() < 0) return nullptr;
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i)
        if (at(i)->op && !std::strcmp(at(i)->op, op)) return at(i);
    return nullptr;
}

static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: return 4;
        case RAD_BF16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

struct Opd {
    uint32_t dt = RAD_BF16;
    std::vector<int64_t> shape;
    int64_t pitch = 0;
    std::vector<uint8_t> bytes;
    bool absent = false, out = false;
    double rtol = 0, atol = 0;
    int64_t elements() const {
        if (shape.size() == 2 && pitch) return shape[0] * pitch;
        int64_t n = 1;
        for (int64_t e : shape) n *= e;
        return n;
    }
    RadTensor tensor(void* data) const {
        RadTensor t{};
        t.data = absent ? nullptr : data;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[(size_t)i];
            t.stride[i] = st;
            st *= shape[(size_t)i];
        }
        if (shape.size() == 2 && pitch) t.stride[0] = pitch;
        return t;
    }
};
static Opd opd(uint32_t dt, std::vector<int64_t> shape, int64_t pitch = 0) {
    Opd o;
    o.dt = dt;
    o.shape = std::move(shape);
    o.pitch = pitch;
    o.bytes.assign((size_t)o.elements() * esz(dt), 0x7e);
    return o;
}
static Opd absent() { Opd o; o.absent = true; return o; }

static std::vector<int64_t> offsets(const RadTensor& t) {
    std::vector<int64_t> o;
    const int64_t d0 = t.shape[0], d1 = t.rank > 1 ? t.shape[1] : 1, d2 = t.rank > 2 ? t.shape[2] : 1;
    for (int64_t i = 0; i < d0; ++i)
        for (int64_t j = 0; j < d1; ++j)
            for (int64_t k = 0; k < d2; ++k)
                o.push_back(i * t.stride[0] + (t.rank > 1 ? j * t.stride[1] : 0) +
                            (t.rank > 2 ? k * t.stride[2] : 0));
    return o;
}
static void fill(Opd& o, const std::function<double(int64_t)>& f) {
    const RadTensor t = o.tensor(o.bytes.data());
    int64_t idx = 0;
    for (int64_t off : offsets(t)) {
        const double v = f(idx++);
        switch (o.dt) {
            case RAD_BF16: ((uint16_t*)o.bytes.data())[off] = f2b((float)v); break;
            case RAD_F32: ((float*)o.bytes.data())[off] = (float)v; break;
            case RAD_I32: ((int32_t*)o.bytes.data())[off] = (int32_t)v; break;
            case RAD_I64: ((int64_t*)o.bytes.data())[off] = (int64_t)v; break;
            default: break;
        }
    }
}
static double get(uint32_t dt, const uint8_t* p, int64_t off) {
    switch (dt) {
        case RAD_BF16: return b2f(((const uint16_t*)p)[off]);
        case RAD_F32: return ((const float*)p)[off];
        case RAD_I32: return ((const int32_t*)p)[off];
        case RAD_I64: return (double)((const int64_t*)p)[off];
        default: return ((const uint8_t*)p)[off];
    }
}

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float u() { return (float)(next() >> 8) / 8388608.0f - 1.0f; }
    float n() { return (u() + u() + u() + u()) * 0.866f; }
    uint32_t r(uint32_t m) { return next() % m; }
};

static RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

static int g_fails = 0, g_cases = 0;

static void check(const char* op, RadLaunchFn kernel, const std::string& what, std::vector<RadParam> p,
                  std::vector<Opd>& ops) {
    ++g_cases;
    const RadKernelInfo* ref = libref_op(op);
    if (!ref) {
        std::printf("FAIL %s: libref's %s not loadable (set RADIANCE_HOME)\n", what.c_str(), op);
        ++g_fails;
        return;
    }
    const size_t n = ops.size();
    std::vector<std::vector<uint8_t>> a(n), b(n);
    std::vector<RadTensor> ta(n), tb(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = b[i] = ops[i].bytes;
        ta[i] = ops[i].tensor(a[i].data());
        tb[i] = ops[i].tensor(b[i].data());
    }
    RadArgs ar{}, ak{};
    ar.t = ta.data(); ar.n_t = (int)n; ar.p = p.data(); ar.n_p = (int)p.size(); ar.world_size = 1;
    ak.t = tb.data(); ak.n_t = (int)n; ak.p = p.data(); ak.n_p = (int)p.size(); ak.world_size = 1;
    const int rr = ref->launch(&ar, nullptr), rk = kernel(&ak, nullptr);
    if (rr != RAD_OK || rk != RAD_OK) {
        std::printf("FAIL %s: libref rc=%d kernel rc=%d\n", what.c_str(), rr, rk);
        ++g_fails;
        return;
    }
    int64_t bad = 0, total = 0;
    bool padding = true;
    for (size_t i = 0; i < n; ++i) {
        if (ops[i].absent) continue;
        std::vector<char> live(a[i].size() / esz(ops[i].dt), 0);
        for (int64_t off : offsets(ta[i])) {
            live[(size_t)off] = 1;
            if (!ops[i].out) continue;
            const double e = get(ops[i].dt, a[i].data(), off), g = get(ops[i].dt, b[i].data(), off);
            ++total;
            if (!std::isfinite(g) || std::fabs(g - e) > ops[i].atol + ops[i].rtol * std::fabs(e)) {
                if (!bad) std::printf("  first: operand %zu element %lld got %.9g want %.9g\n", i, (long long)off, g, e);
                ++bad;
            }
        }
        const size_t es = esz(ops[i].dt);
        for (size_t off = 0; off < live.size(); ++off)
            if (!live[off] && std::memcmp(&b[i][off * es], &ops[i].bytes[off * es], es)) padding = false;
        if (!ops[i].out && std::memcmp(b[i].data(), ops[i].bytes.data(), b[i].size())) padding = false;
    }
    if (bad || !padding || !total) {
        std::printf("FAIL %s: %lld of %lld out of bound%s\n", what.c_str(), (long long)bad,
                    (long long)total, padding ? "" : ", padding or an input written");
        ++g_fails;
    } else {
        std::printf("ok   %s (%lld values)\n", what.c_str(), (long long)total);
    }
}

static const double kUlp = 0.0079;   /* one bf16 ulp, relative */

static void ihc_pre() {
    struct G { int64_t hc, n, M, pad; bool post; };
    for (const G& g : { G{ 4, 96, 3, 0, true }, G{ 4, 40, 2, 5, true }, G{ 2, 17, 3, 0, true },
                        G{ 3, 64, 2, 0, false }, G{ 8, 24, 2, 3, true } }) {
        Rng rng((uint32_t)(g.hc * 100 + g.n));
        const int64_t W = g.hc * g.n, rows = g.post ? 2 * g.hc : g.hc;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 2; });
        ops.push_back(opd(RAD_F32, { rows, W }));
        fill(ops[1], [&](int64_t) { return rng.n() * 0.05; });
        ops.push_back(opd(RAD_F32, { g.post ? 2 : 1 }));
        fill(ops[2], [&](int64_t i) { return i ? 2.1 : 3.0; });
        ops.push_back(opd(RAD_F32, { rows }));
        fill(ops[3], [&](int64_t) { return rng.n() * 0.7; });
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        ops[4].out = true; ops[4].rtol = kUlp; ops[4].atol = 1e-5;
        if (g.post) {
            ops.push_back(opd(RAD_F32, { g.M, g.hc }, g.pad ? g.hc + 2 : 0));
            ops[5].out = true; ops[5].rtol = 2e-4; ops[5].atol = 1e-6;
        } else {
            ops.push_back(absent());
        }
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pi("rows", rows),
                                    pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
        if (g.post) p.push_back(pf("post_mag", 2.0));
        char w[120];
        std::snprintf(w, sizeof w, "ihc_pre hc %lld n %lld M %lld%s%s", (long long)g.hc, (long long)g.n,
                      (long long)g.M, g.post ? "" : " head", g.pad ? " pitched" : "");
        check("ihc_pre", r4d_rad_ihc_pre, w, p, ops);
    }
}

static void ihc_post() {
    struct G { int64_t hc, n, M, pad; };
    for (const G& g : { G{ 4, 64, 3, 0 }, G{ 4, 64, 2, 8 }, G{ 2, 17, 3, 0 }, G{ 3, 24, 2, 5 } }) {
        Rng rng((uint32_t)(g.hc * 10 + g.n));
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 3; });
        ops.push_back(opd(RAD_F32, { g.M, g.hc }));
        fill(ops[1], [&](int64_t) { return 0.5 + rng.u(); });
        ops.push_back(opd(RAD_BF16, { g.M, g.hc * g.n }, g.pad ? g.hc * g.n + g.pad : 0));
        fill(ops[2], [&](int64_t) { return rng.n() * 2; });
        ops[2].out = true;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), ps("dtype", "bf16") };
        char w[120];
        std::snprintf(w, sizeof w, "ihc_post hc %lld n %lld M %lld%s, bit for bit", (long long)g.hc,
                      (long long)g.n, (long long)g.M, g.pad ? " pitched" : "");
        check("ihc_post", r4d_rad_ihc_post, w, p, ops);
    }
}

static void gate_mul() {
    struct G { int64_t M, n, pad; };
    for (const G& g : { G{ 3, 64, 0 }, G{ 2, 80, 8 }, G{ 3, 13, 0 }, G{ 2, 37, 3 } }) {
        Rng rng((uint32_t)(g.n * 3 + g.M));
        std::vector<Opd> ops;
        for (int i = 0; i < 2; ++i) {
            ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
            fill(ops.back(), [&](int64_t) { return rng.n() * (i ? 4 : 2); });
        }
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 1e-30;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "gate_mul M %lld n %lld%s", (long long)g.M, (long long)g.n, g.pad ? " pitched" : "");
        check("gate_mul", r4d_rad_gate_mul, w, p, ops);
    }
}

static void mla_kv() {
    for (const char* mode : { "absorb", "expand" })
        for (int64_t M : { 1, 3, 9 }) {
            const bool ab = !std::strcmp(mode, "absorb");
            const int64_t H = 2, kd = 192, vd = 256, L = 512, xw = ab ? kd : L, yw = ab ? L : vd;
            Rng rng((uint32_t)(M + (ab ? 7 : 70)));
            std::vector<Opd> ops;
            ops.push_back(opd(RAD_BF16, { M, H, xw }));
            fill(ops[0], [&](int64_t) { return rng.n(); });
            ops.push_back(opd(RAD_BF16, { H * (kd + vd), L }));
            fill(ops[1], [&](int64_t) { return rng.n() * 0.05; });
            ops.push_back(opd(RAD_BF16, { M, H, yw }));
            ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 2e-5;
            std::vector<RadParam> p = { pi("M", M), pi("heads", H), pi("k_dim", kd), pi("v_dim", vd),
                                        pi("latent_dim", L), ps("mode", mode), ps("dtype", "bf16") };
            char w[100];
            std::snprintf(w, sizeof w, "mla_project_kv %s heads %lld M %lld", mode, (long long)H, (long long)M);
            check("mla_project_kv", r4d_rad_mla_project_kv, w, p, ops);
        }
    struct G { int64_t M, H, kd, vd, rd, pad; };
    for (const G& g : { G{ 3, 2, 16, 24, 8, 0 }, G{ 2, 3, 16, 8, 8, 8 }, G{ 3, 2, 13, 10, 6, 0 }, G{ 2, 2, 12, 9, 5, 3 } }) {
        Rng rng((uint32_t)(g.M * 7 + g.kd));
        const int64_t bw = g.H * (g.kd + g.vd), kw = g.H * (g.kd + g.rd), vw = g.H * g.vd;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, bw }, g.pad ? bw + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_BF16, { g.M, g.rd }, g.pad ? g.rd + g.pad : 0));
        fill(ops[1], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_BF16, { g.M, kw }, g.pad ? kw + g.pad : 0));
        ops.push_back(opd(RAD_BF16, { g.M, vw }, g.pad ? vw + g.pad : 0));
        ops[2].out = ops[3].out = true;
        std::vector<RadParam> p = { pi("M", g.M), pi("heads", g.H), pi("k_dim", g.kd), pi("v_dim", g.vd),
                                    pi("rope_dim", g.rd), ps("dtype", "bf16") };
        char w[120];
        std::snprintf(w, sizeof w, "mla_kv_split %lld/%lld/%lld heads %lld%s, bit for bit", (long long)g.kd,
                      (long long)g.vd, (long long)g.rd, (long long)g.H, g.pad ? " pitched" : "");
        check("mla_kv_split", r4d_rad_mla_kv_split, w, p, ops);
    }
}

static void engram_ids() {
    const int64_t heads = 4, ngram = 3, pad = 90, msize = 100, image = 95, video = 96;
    for (int form = 0; form < 4; ++form) {
        const bool decode = form & 1, wide = form & 2;
        const int64_t S = 3, W = decode ? 4 : 2;
        const std::vector<int64_t> Ts = decode ? std::vector<int64_t>{ 3, 1, 2 } : std::vector<int64_t>{ 4, 1, 3 };
        int64_t M = 0;
        for (int64_t t : Ts) M += t;
        Rng rng((uint32_t)(form + 3));
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_I32, { M }));
        fill(ops[0], [&](int64_t) {
            const uint32_t r = rng.r(12);
            return r == 0 ? (double)pad : r == 1 ? (double)image : r == 2 ? (double)video : r == 3 ? 150.0 : (double)rng.r(90);
        });
        ops.push_back(opd(RAD_I32, { 4, W }));
        fill(ops[1], [&](int64_t) { return (double)rng.r(90); });
        ops[1].out = true;
        std::vector<int> cu = { 0 };
        for (int64_t t : Ts) cu.push_back(cu.back() + (int)t);
        ops.push_back(opd(RAD_I32, { S + 1 }));
        fill(ops[2], [&](int64_t i) { return cu[(size_t)i]; });
        ops.push_back(opd(RAD_I32, { M }));
        {
            int64_t i = 0;
            for (int64_t s = 0; s < S; ++s)
                for (int64_t t = 0; t < Ts[(size_t)s]; ++t, ++i)
                    ((int32_t*)ops[3].bytes.data())[i] = (int32_t)((s == 0 ? 0 : 500) + t);
        }
        ops.push_back(opd(RAD_I32, { msize }));
        fill(ops[4], [&](int64_t) { return (double)rng.r(60); });
        ops.push_back(opd(RAD_I64, { ngram }));
        { const int64_t m[3] = { 1000003, 7919000117ll, 0x5851F42D4C957F2Dll };
          for (int i = 0; i < 3; ++i) ((int64_t*)ops[5].bytes.data())[i] = m[i]; }
        ops.push_back(opd(RAD_I64, { heads }));
        fill(ops[6], [&](int64_t i) { return (double)(1009 + 13 * i); });
        ops.push_back(opd(RAD_I64, { heads }));
        fill(ops[7], [&](int64_t i) { return (double)(i * 2003); });
        if (decode) {
            ops.push_back(opd(RAD_I32, { S }));
            fill(ops[8], [&](int64_t i) { return (double)(1 + i % 3); });
        } else {
            ops.push_back(absent());
        }
        ops.push_back(opd(RAD_I32, { S }));
        { const int32_t slots[3] = { 3, -1, 0 };
          fill(ops[9], [&](int64_t i) { return (double)slots[i]; }); }
        ops.push_back(opd(wide ? RAD_I64 : RAD_I32, { M, heads }));
        ops.push_back(opd(RAD_I32, { M }));
        ops[10].out = ops[11].out = true;
        std::vector<RadParam> p = { pi("M", M), pi("heads", heads), pi("ngram", ngram), pi("pad", pad),
                                    pi("map_size", msize), pi("image", image), pi("video", video) };
        char w[100];
        std::snprintf(w, sizeof w, "engram_ids %s, ids i%d, bit for bit", decode ? "decode" : "prefill", wide ? 64 : 32);
        check("engram_ids", r4d_rad_engram_ids, w, p, ops);
    }
}

static void engram_norm_gate() {
    struct N { int64_t M, heads, n, pad; double eps; };
    for (const N& g : { N{ 3, 4, 24, 0, 0.0078125 }, N{ 2, 3, 37, 5, 1.19e-7 } }) {
        Rng rng((uint32_t)(g.n));
        const int64_t W = g.heads * g.n;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 3; });
        ops.push_back(opd(RAD_BF16, { W }));
        fill(ops[1], [&](int64_t) { return 1 + rng.n() * 0.3; });
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 1e-30;
        std::vector<RadParam> p = { pi("M", g.M), pi("heads", g.heads), pi("n", g.n), pf("eps", g.eps),
                                    ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "engram_norm heads %lld n %lld eps %g", (long long)g.heads, (long long)g.n, g.eps);
        check("engram_norm", r4d_rad_engram_norm, w, p, ops);
    }
    struct G { int64_t M, hc, n, pad; double eps; };
    for (const G& g : { G{ 4, 2, 48, 0, 0.0078125 }, G{ 3, 4, 20, 6, 1.19e-7 }, G{ 3, 1, 33, 0, 1e-6 } }) {
        Rng rng((uint32_t)(g.n * 5));
        const int64_t W = g.hc * g.n;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t i) { return i / W == 1 ? 0.0 : rng.n() * 2; });   /* row 1: d == 0 */
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        fill(ops[1], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_I32, { g.M }));
        fill(ops[2], [&](int64_t i) { return i % 3 == 2 ? 0.0 : 1.0; });
        for (int k = 0; k < 4; ++k) {
            ops.push_back(opd(RAD_BF16, { k == 2 ? g.n : W }));
            fill(ops.back(), [&](int64_t) { return 1 + rng.n() * 0.2; });
        }
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[7], [&](int64_t) { return rng.n() * 2; });
        ops[7].out = true; ops[7].rtol = 2 * kUlp; ops[7].atol = 0.016;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pf("eps", g.eps), ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "engram_gate hc %lld n %lld eps %g%s", (long long)g.hc, (long long)g.n, g.eps,
                      g.pad ? " pitched" : "");
        check("engram_gate", r4d_rad_engram_gate, w, p, ops);
    }
}

int main() {
    ihc_pre();
    ihc_post();
    gate_mul();
    mla_kv();
    engram_ids();
    engram_norm_gate();
    std::printf("%d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
