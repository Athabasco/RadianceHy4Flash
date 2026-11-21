/* hyv4flash_common.h -- what every block of the Hy4-flash graph shares: the geometry read from the
 * config, the op and weight helpers, the debug taps and the independent hyper-connection.
 *
 * Included by hyv4flash.cpp only (the blocks are header-only, in one translation unit). */
#ifndef HYV4FLASH_COMMON_H
#define HYV4FLASH_COMMON_H

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace hyv4flash {
using namespace rad::arch;

/* One attention geometry: the sparse layers', the sliding layers', or the drafter's. */
struct AttnGeom {
    int64_t heads = 0, qk_nope = 0, qk_rope = 0, v = 0;
    int64_t q_width() const { return heads * (qk_nope + qk_rope); }
    int64_t kv_width() const { return heads * (qk_nope + v); }
    int64_t v_width() const { return heads * v; }
    int64_t qk() const { return qk_nope + qk_rope; }
};

struct Shape {
    int64_t n_layer = 0, n_embd = 0, n_vocab = 0, hc = 0;
    int64_t q_lora = 0, kv_lora = 0;
    AttnGeom sparse{}, sliding{};
    std::vector<int> is_sliding;   /* per layer */
    std::vector<int> is_moe;       /* per layer */
    /* feed-forward */
    int64_t n_ff = 0, n_expert = 0, n_ff_exp = 0, latent = 0, n_ff_shared = 0, top_k = 0;
    double route_scale = 1.0, swiglu_limit = 0.0;
    int norm_topk = 1;
    /* the compressed indexer: heads x dim, the pooling ratio, the groups a query recalls
     * (group_topk - 1 closed ones plus its own), and the context at or below which a sparse
     * layer attends densely (group_topk * ratio, HF's `candidate_topk`) */
    int64_t ix_heads = 0, ix_dim = 0, ix_pool = 0, ix_group_topk = 0, dense_limit = 0;
    /* Engram: `layers` are layer indices (the config's ids are one above them) */
    std::vector<int> engram_layers;
    std::vector<int64_t> engram_rows;   /* each layer's table, the sum of its head moduli */
    int64_t engram_ngram = 0, engram_heads = 0, engram_head_dim = 0, engram_map = 0;
    int64_t engram_pad = 0, image_token = -1, video_token = -1;
    bool engram_fp8 = false;
    /* hyper-connections, rotary, window */
    double hc_eps = 1e-6, hc_mag = 2.0, eps = 1e-6;
    double theta_sparse = 1e7, theta_sliding = 1e4;
    int64_t window = 0;   /* keys a sliding row sees, its own token included */
    /* DFly2 and the tower */
    int64_t dfly_layers = 0;
    int64_t v_layers = 0, v_embd = 0, v_ff = 0, v_patch = 0, v_chan = 0, v_merge = 0, v_pos = 0;

    /* This rank's share (PLAN §1.5, set by split_ranks; at one rank the whole). The attention's
     * heads are whole and contiguous a rank (q_nope, q_rope, kv_b, attn_gate and the sinks by
     * rows, attn_output by columns); the dense MLP and the shared expert split their width evenly;
     * the routed experts are either whole a rank (`experts_whole`: experts [expert_base,
     * expert_base + n_expert_r), for bf16 experts and calibration) or a slice of every expert's
     * width (n_ff_exp_r, the quantised serving form). Everything else is replicated. The geometry
     * above stays the model's: the drafter and the tower are declared whole. */
    int world = 1, rank = 0;
    AttnGeom sparse_r{}, sliding_r{};
    int64_t n_ff_r = 0, n_ff_shared_r = 0, n_expert_r = 0, expert_base = 0, n_ff_exp_r = 0;
    bool experts_whole = true;
    const AttnGeom& local(bool sliding_layer) const { return sliding_layer ? sliding_r : sparse_r; }
};

inline int refuse(const char* what) {
    fprintf(stderr, "radiance: hyv4flash: %s\n", what);
    return RAD_E_UNSUPPORTED;
}

/* An op with its liveness, from lists: the RAD_OP macros take fixed arrays, and most ops here are
 * declared from a loop. Returns 0 where no library serves it. */
inline rad_op mk(RadBuilder* b, const char* name, std::initializer_list<RadParam> ps,
                 std::initializer_list<rad_weight> ws, std::initializer_list<rad_buf> reads,
                 std::initializer_list<rad_buf> writes) {
    const rad_op h = rad_decl_op(b, name, ps.begin(), (int)ps.size(),
                                 ws.size() ? ws.begin() : nullptr, (int)ws.size());
    return rw(b, h, reads, writes);
}

/* An op that must resolve: a graph with a hole would look servable. */
#define HY4_NEED(h, what)                                                                      \
    do {                                                                                       \
        if (!(h)) {                                                                            \
            fprintf(stderr, "radiance: hyv4flash: no loaded library serves %s\n", (what));     \
            return RAD_E_INVAL;                                                                \
        }                                                                                      \
    } while (0)

/* A weight the reference keeps in f32 (a norm's gain, a hyper-connection table, a sink, the
 * indexer's pooling, the router's correction): widened from bf16 at load where the checkpoint
 * stores it so. A recipe that quantised one is refused by name: nothing here reads its codes. */
inline rad_weight decl_f32(RadBuilder* b, const char* name, const char* src,
                           std::initializer_list<int64_t> shape, RadWeightGroup grp,
                           int access = RAD_ACCESS_PER_TOKEN, int shard = RAD_SHARD_NONE) {
    if (map_copy(b, name, src) != RAD_OK) return 0;
    RadEncoding e{};
    if (weight_enc(b, name, &e) && !(rad_enc_is(&e, "plain") && e.n_planes == 1)) {
        char en[256];
        rad_enc_format(&e, en, sizeof en);
        fprintf(stderr, "radiance: hyv4flash: '%s' is %s; it is a parameter the model applies "
                        "in f32, which this plugin reads only as stored (bf16 or f32). Convert it "
                        "without a recipe rule.\n", name, en);
        return 0;
    }
    return decl_w(b, name, RAD_F32, shape, access, shard, grp);
}

/* The all-reduce of a rank's partial [rows, n] in place, before a norm that needs the full sum:
 * exact bf16 (summed in f32 in rank order, rounded once), whatever wire the deployment allows,
 * because the norm behind it and the calibration read the sum. 0 at one rank. At more ranks a
 * library that serves no exact reduction is a refusal (PLAN §1.5), returned in `*rc`. */
inline rad_op decl_all_reduce(RadBuilder* b, const Geom& g, int64_t n, rad_buf x, int* rc) {
    *rc = RAD_OK;
    if (g.world <= 1) return 0;
    const rad_op h = mk(b, "all_reduce",
                        { RAD_INT("world_size", g.world), RAD_RANGE("numel", n, g.max_tok * n),
                          RAD_STR("dtype", "bf16"), RAD_INT("exact", 1),
                          RAD_INT("min_bytes", g.wire_min_bytes) },
                        {}, { x }, { x });
    if (!h) {
        fprintf(stderr, "radiance: hyv4flash: no loaded library serves an exact bf16 all_reduce "
                        "over %d ranks; the tensor-parallel graph needs one before each sandwich "
                        "norm\n", g.world);
        *rc = RAD_E_UNSUPPORTED;
    }
    return h;
}
inline void issue_all_reduce(RadCtx* c, rad_op op, rad_buf x, int64_t rows, int64_t n) {
    if (op && rows > 0) RAD_ISSUE_N(c, op, rows * n, brows(x, rows), RAD_NONE);
}

/* A plain bf16 weight that an op other than a linear reads (kv_b, the Engram table's bf16 form). */
inline bool is_plain_bf16(RadBuilder* b, const char* name) {
    RadEncoding e{};
    if (!weight_enc(b, name, &e)) return true;   /* no model behind the builder: as declared */
    return rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16;
}

/* `rmsnorm` over `n` columns with an f32 gain. */
inline rad_op decl_rmsnorm(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight gain,
                           rad_buf in, rad_buf out) {
    return mk(b, "rmsnorm",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                RAD_STR("dtype", "bf16") },
              { gain }, { in }, { out });
}

/* A copy, `cast` to the same dtype: views to and from strided column slices. */
inline const char* dt_name(uint32_t dt) {
    return dt == RAD_F32 ? "f32" : dt == RAD_I32 ? "i32" : "bf16";
}
inline rad_op decl_copy(RadBuilder* b, int64_t rows, int64_t n, uint32_t dt, rad_buf in,
                        rad_buf out) {
    return mk(b, "cast",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("from", dt_name(dt)),
                RAD_STR("to", dt_name(dt)) },
              {}, { in }, { out });
}

/* rope (interleaved pairs, `gptj`) over `heads` heads of `dim`, all of them rotated, in place.
 * The text trunk's M-RoPE with equal components is this rotation (plan D6). */
inline rad_op decl_rope_gptj(RadBuilder* b, int64_t rows, int64_t heads, int64_t dim,
                             double theta, rad_buf x) {
    return mk(b, "rope",
              { RAD_RANGE("M", 1, rows), RAD_INT("head_dim", dim), RAD_INT("n_head", heads),
                RAD_INT("n_head_kv", 0), RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                RAD_STR("mode", "gptj"), RAD_INT("rotary_dim", dim) },
              {}, { x }, { x });
}

/* ------------------------------------------------------------------ debug taps
 * RADIANCE_HY4_TAPS=1 (the harness, ref/README.md "Tap contract"): a PERSIST buffer
 * `dbg.<key>.<l>` (`dbg.<key>` for a global key) a tap, [max_tok, width] at the producer's dtype,
 * and a copy into it behind the producing op. Read back through the engine's buffer probe
 * (RADIANCE_DUMP_BUF / RADIANCE_DUMP_DIR). Off, nothing is declared. */
inline bool taps_on() {
    const char* e = std::getenv("RADIANCE_HY4_TAPS");
    return e && *e && std::strcmp(e, "0") != 0;
}

struct Tap {
    rad_buf buf = 0;
    rad_op op = 0;
    int64_t width = 0;
    explicit operator bool() const { return op != 0; }
    /* `src` read [rows, width]; `width` 0 takes the tap's. */
    int declare(RadBuilder* b, Names& nm, int64_t max_rows, const char* key, int layer,
                uint32_t dt, int64_t w, rad_buf src) {
        if (!taps_on()) return RAD_OK;
        width = w;
        const char* name = layer < 0 ? nm.f("dbg.%s", key) : nm.f("dbg.%s.%d", key, layer);
        buf = decl_b(b, name, dt, { max_rows, w }, RAD_BUF_PERSIST);
        if (!buf) return RAD_E_INVAL;
        op = decl_copy(b, max_rows, w, dt, src, buf);
        HY4_NEED(op, "the debug taps' copy (cast)");
        return RAD_OK;
    }
    void step(RadCtx* c, RadOperand src, int64_t rows) const {
        if (op && rows > 0) RAD_ISSUE_N(c, op, rows, src, brows(buf, rows));
    }
};

/* ------------------------------------------------------------------ hyper-connections
 * The independent hyper-connection (HYV4FlashHyperConnection): `fn` scores the flattened streams
 * into 2 * hc logits (the read gate's hc, then the write gate's), `base` their biases, `scale` one
 * factor for each half; ihc_pre collapses the streams into the block's input and gives the write
 * weights, ihc_post adds the block's output back into every stream. f32 in the checkpoint and in
 * the reference. `head` is the final collapse (hc rows, one scale, no write gate). */
struct Hyper {
    rad_weight fn = 0, base = 0, scale = 0;
    rad_op op_pre = 0, op_post = 0;
    bool head = false;
    int declare(RadBuilder* b, Names& nm, const Shape& s, int64_t max_tok, const char* scope,
                const char* fn_src, const char* base_src, const char* scale_src,
                RadWeightGroup grp, bool is_head, rad_buf h, rad_buf x, rad_buf post) {
        head = is_head;
        const int64_t rows = head ? s.hc : 2 * s.hc;
        fn = decl_f32(b, nm.f("%s.fn", scope), fn_src, { rows, s.hc * s.n_embd }, grp);
        base = decl_f32(b, nm.f("%s.base", scope), base_src, { rows }, grp);
        scale = decl_f32(b, nm.f("%s.scale", scope), scale_src, { head ? 1 : 2 }, grp);
        if (!fn || !base || !scale) return RAD_E_INVAL;
        if (head) {
            op_pre = mk(b, "ihc_pre",
                        { RAD_RANGE("M", 1, max_tok), RAD_INT("n", s.n_embd), RAD_INT("hc", s.hc),
                          RAD_INT("rows", rows), RAD_F64("eps", s.eps), RAD_F64("hc_eps", s.hc_eps),
                          RAD_STR("dtype", "bf16") },
                        { fn, scale, base }, { h }, { x });
        } else {
            op_pre = mk(b, "ihc_pre",
                        { RAD_RANGE("M", 1, max_tok), RAD_INT("n", s.n_embd), RAD_INT("hc", s.hc),
                          RAD_INT("rows", rows), RAD_F64("eps", s.eps), RAD_F64("hc_eps", s.hc_eps),
                          RAD_F64("post_mag", s.hc_mag), RAD_STR("dtype", "bf16") },
                        { fn, scale, base }, { h }, { x, post });
        }
        HY4_NEED(op_pre, "ihc_pre");
        return RAD_OK;
    }
    /* The write, declared where it is issued (declaration order is liveness): behind the block. */
    int declare_post(RadBuilder* b, const Shape& s, int64_t max_tok, rad_buf h, rad_buf post,
                     rad_buf y) {
        op_post = mk(b, "ihc_post",
                     { RAD_RANGE("M", 1, max_tok), RAD_INT("n", s.n_embd), RAD_INT("hc", s.hc),
                       RAD_STR("dtype", "bf16") },
                     {}, { y, post, h }, { h });
        HY4_NEED(op_post, "ihc_post");
        return RAD_OK;
    }
    void pre(RadCtx* c, rad_buf h, rad_buf x, rad_buf post, int64_t T) const {
        if (head)
            RAD_ISSUE_N(c, op_pre, T, brows(h, T), RAD_W(fn), RAD_W(scale), RAD_W(base),
                        brows(x, T), RAD_NONE);
        else
            RAD_ISSUE_N(c, op_pre, T, brows(h, T), RAD_W(fn), RAD_W(scale), RAD_W(base),
                        brows(x, T), brows(post, T));
    }
    void post_add(RadCtx* c, rad_buf y, rad_buf post, rad_buf h, int64_t T) const {
        RAD_ISSUE_N(c, op_post, T, brows(y, T), brows(post, T), brows(h, T));
    }
};

/* The buffers every layer shares: the streams, the blocks' inputs and outputs. */
struct Trunk {
    rad_buf h = 0;        /* [T, hc*n] the streams */
    rad_buf x = 0;        /* [T, n] a connection's collapse */
    rad_buf post = 0;     /* [T, hc] f32 its write weights */
    rad_buf xn = 0;       /* [T, n] the block's normed input */
    rad_buf y = 0;        /* [T, n] the block's normed output */
};

}  // namespace hyv4flash
#endif
