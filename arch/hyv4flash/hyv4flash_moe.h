/* hyv4flash_moe.h -- Hy4-flash's feed-forward: layer 0's dense MLP and the latent MoE of the rest.
 *
 *   dense   y = rmsnorm(down(silu(gate(x)) * up(x)))                       (post_fc2_norm)
 *   MoE     the router scores x in f32 (sigmoid, the correction bias for the selection only,
 *           top-k normalised, times routed_scaling_factor); the routed experts run in the latent
 *           (routed_expert_down_proj in front, routed_expert_up_proj behind) with the clamped
 *           SwiGLU (gate <= limit, |up| <= limit); the shared expert reads x and is unclamped;
 *           y = rmsnorm(routed + shared)                                    (combine_output_norm)
 *
 * The routed arm is MoeFP8 at the latent's width (nemotronh's LatentMoe arrangement): its input is
 * latent_in's output, its router reads x over Config::router_k, and its gather writes the routed
 * sum back into the latent plane, which latent_out widens. Its router is GlmRouter's, so it
 * declares the router as `blk.<l>.glm_router.weight` / `.correction` (PLAN D10). */
#ifndef HYV4FLASH_MOE_H
#define HYV4FLASH_MOE_H

#include <arch/rad_block_moe_fp8.h>

#include "hyv4flash_common.h"

namespace hyv4flash {

/* A gated MLP over `x`: gate and up fused (gate rows first), SiLU, down. `ff_w` is this rank's
 * share of the width: gate_up's rows of each half (RAD_SHARD_ROW over two row parts), down's
 * columns (RAD_SHARD_COL); its output is the rank's partial sum. */
struct GatedMlp {
    Lin gate_up{}, down{};
    GateQuantFP8 act{};
    rad_buf gu = 0;
    ActFP8 ff{};
    int declare(RadBuilder* b, Names& nm, const Geom& g, const Shape& s, int l, const char* gu_name,
                const char* dn_name, const char* mp, int64_t ff_w, rad_buf x, rad_buf y,
                rad_buf gu_buf, rad_buf ff_buf) {
        gu = gu_buf;
        ff.x = ff_buf;
        ActFP8 in{};
        in.x = x;
        const auto grp = grp_layer(l);
        RAD_ARCH_TRY(gate_up.declare(b, nm, g, gu_name, 2 * ff_w, s.n_embd, RAD_SHARD_ROW, grp,
                                     { nm.ckpt("%s.gate_proj.weight", mp),
                                       nm.ckpt("%s.up_proj.weight", mp) },
                                     in, gu, 0, { ff_w, ff_w }));
        RAD_ARCH_TRY(act.declare(b, g, gu, ff, ff_w, "silu"));
        RAD_ARCH_TRY(down.declare(b, nm, g, dn_name, s.n_embd, ff_w, RAD_SHARD_COL, grp,
                                  { nm.ckpt("%s.down_proj.weight", mp) }, ff, y));
        return RAD_OK;
    }
    void step(RadCtx* c, rad_buf x, rad_buf y, int64_t T) const {
        ActFP8 in{};
        in.x = x;
        gate_up.step(c, in, gu, T);
        act.step(c, gu, T);
        down.step(c, ff, y, T);
    }
};

/* The routed arm's activation codes (MoeWire::declare). */
enum { kArmBf16 = 0, kArmE4M3 = 1, kArmI8 = 2 };

/* The routed arm's planes and the shared arm's, shared by every MoE layer. */
struct MoeWire {
    ActFP8 lat{};          /* [T, latent] latent_in's output, then the routed sum */
    rad_buf routed = 0;    /* [T, n] latent_out's output, then routed + shared */
    rad_buf sgu = 0;       /* [T, 2 * ff_shared] */
    rad_buf sff = 0;       /* [T, ff_shared] */
    rad_buf sout = 0;      /* [T, n] */
    MoeFP8::Wire moe{};
    /* The routed arm's activation codes, by the experts' format (Ffn::arm): none for bf16
     * experts; for the codebook forms (w4nl64a8h, w5nl64a8h, w6nl64a8h) the rotated E4M3 pairs of
     * the latent (`hr`, had_quant_act_fp8) and of the clamped product (`eff`); for int8 and IQR
     * experts their rotated int8 pairs (had_quant_act_i8g), with the product's E4M3 pair beside
     * it for a codebook down projection behind a trellis gate_up. The latent's own E4M3 pair is
     * declared for MoeFP8's prepared-input check and read by nothing: the routed GEMMs read `hr`. */
    int declare(RadBuilder* b, Names& nm, const Shape& s, const Geom& g, int arm) {
        /* The rank's experts and width (whole experts: n_expert_r of the full width; sliced: all
         * of them, n_ff_exp_r wide); the router scores all of them. */
        const int64_t T = g.max_tok, k = s.top_k, F = s.n_ff_exp_r, L = s.latent;
        auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
            return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
        };
        lat.x = bf("moe.latent", { T, L });
        routed = bf("moe.routed", { T, s.n_embd });
        sgu = bf("moe.shared_gate_up", { T, 2 * s.n_ff_shared_r });
        sff = bf("moe.shared_ff", { T, s.n_ff_shared_r });
        sout = bf("moe.shared_out", { T, s.n_embd });
        auto& w = moe;
        w.logits = decl_b(b, nm.f("moe.router_logits"), RAD_F32, { T, s.n_expert });
        w.ids = decl_b(b, nm.f("moe.expert_ids"), RAD_I32, { T, k });
        w.ew = decl_b(b, nm.f("moe.expert_weights"), RAD_F32, { T, k });
        w.eoff = decl_b(b, nm.f("moe.expert_offsets"), RAD_I32, { s.n_expert_r + 1 });
        w.ecnt = decl_b(b, nm.f("moe.expert_counts"), RAD_I32, { s.n_expert_r });
        w.sorted = decl_b(b, nm.f("moe.expert_sorted"), RAD_I32, { T * k });
        w.egu = bf("moe.expert_gate_up", { T * k, 2 * F });
        w.eff.x = bf("moe.expert_ff", { T * k, F });
        w.edn = bf("moe.expert_down", { T * k, L });
        if (arm != kArmBf16) {
            Geom gq = g;
            gq.w_bf16 = false;
            const bool i8 = arm == kArmI8;
            RAD_ARCH_TRY(lat.declare_qs(b, nm, gq, nm.f("moe.latent"), L));
            RAD_ARCH_TRY(w.hr.declare_qs(b, nm, gq, nm.f("moe.latent_rot"), L, 0, i8, !i8));
            w.hr.q8_fed = w.hr.q8_rot = i8;
            RAD_ARCH_TRY(w.eff.declare_qs(b, nm, gq, nm.f("moe.expert_ff"), F, T * k, i8, true));
            w.eff.q8_fed = w.eff.q8_rot = i8;
        }
        return lat.x && routed && sgu && sff && sout && w.logits && w.ids && w.ew && w.eoff &&
                       w.ecnt && w.sorted && w.egu && w.eff.x && w.edn
                   ? RAD_OK
                   : RAD_E_INVAL;
    }
};

struct Ffn {
    bool moe_layer = false;
    Geom g{};
    int layer = 0;
    int64_t n_embd = 0;
    rad_buf xf = 0, yf = 0;
    /* dense */
    GatedMlp mlp{};
    rad_buf mo = 0;
    /* MoE */
    MoeWire w{};
    Lin lat_in{}, lat_out{};
    MoeFP8 moe{};
    GatedMlp shared{};
    rad_op op_join = 0, op_ar = 0;
    rad_weight post_gain = 0;
    rad_op op_post_norm = 0;
    Tap t_logits{}, t_ids{}, t_w{}, t_routed_in{}, t_experts_out{}, t_routed_out{}, t_shared{},
        t_out{};

    /* The routed arm's format, read from layer l's first expert this rank maps before any buffer
     * is declared (Ffn::arm then picks the codes the arm reads). */
    static int probe(RadBuilder* b, Names& nm, const Shape& s, int l, MoeFP8::Src* src) {
        const char* mp = nm.ckpt("model.layers.%d.mlp", l);
        MoeFP8::Config& c = src->cfg;
        c.n_expert = c.n_expert_all = s.n_expert;
        c.top_k = s.top_k;
        c.n_ff_exp = s.n_ff_exp;
        c.n_ff_shared = 0;            /* the shared expert reads x: this file's */
        c.sigmoid_router = 1;
        c.norm_topk = s.norm_topk;
        c.route_scale = s.route_scale;
        c.n_group = 1;
        c.topk_group = 1;
        c.swiglu_limit = s.swiglu_limit;
        c.router_k = s.n_embd;
        c.no_reduce = 1;
        c.calib = calib_dir() != nullptr ? 1 : 0;
        /* The first expert this rank maps (whole experts; the probe reads its format). */
        c.probed_expert = (int64_t)s.rank * (s.n_expert / s.world);
        src->norm = nullptr;          /* the input is the latent, prepared here */
        src->router = nm.ckpt("%s.gate.weight", mp);
        src->router_bias = nm.ckpt("%s.gate.e_score_correction_bias", mp);
        /* The stacked experts: gate_up [E, 2 * ff, latent] gate rows first, down [E, latent, ff]. */
        src->experts_gate_up = nm.ckpt("%s.experts.gate_up_proj", mp);
        src->experts_down = nm.ckpt("%s.experts.down_proj", mp);
        RAD_ARCH_TRY(MoeFP8::probe(b, nm, l, *src, &c));
        return RAD_OK;
    }

    /* The routed experts' placement over the ranks (PLAN §1.5), after the probe: whole experts a
     * rank for bf16 experts (MoeFP8 serves a plain expert whole) and for a calibration run (a
     * Hessian spans an expert's whole K: the friend's TP4 run, 128 of 512 a rank), else a slice
     * of every expert's width in whole 128-column groups (the quantised serving form, 768 / 384 at
     * TP2 / TP4). Every MoE layer has the same experts, so the placement is the model's.
     * `whole`: the caller's choice by the probed format and the calibration switch. */
    static int place(Shape& s, bool whole) {
        s.n_expert_r = s.n_expert;
        s.expert_base = 0;
        s.n_ff_exp_r = s.n_ff_exp;
        s.experts_whole = true;
        if (s.world <= 1) return RAD_OK;
        if (whole) {
            if (s.n_expert % s.world) {
                fprintf(stderr, "radiance: hyv4flash: %lld routed experts do not split whole over "
                                "%d ranks\n", (long long)s.n_expert, s.world);
                return RAD_E_UNSUPPORTED;
            }
            s.n_expert_r = s.n_expert / s.world;
            s.expert_base = (int64_t)s.rank * s.n_expert_r;
            return RAD_OK;
        }
        if (s.n_ff_exp % ((int64_t)RAD_FP8_BLOCK * s.world)) {
            fprintf(stderr, "radiance: hyv4flash: the routed experts' %lld columns do not split "
                            "over %d ranks in whole %d-column groups\n",
                    (long long)s.n_ff_exp, s.world, RAD_FP8_BLOCK);
            return RAD_E_UNSUPPORTED;
        }
        s.n_ff_exp_r = s.n_ff_exp / s.world;
        s.experts_whole = false;
        return RAD_OK;
    }
    /* A layer's routed-arm configuration for the placement. */
    static void apply(const Shape& s, MoeFP8::Config& c) {
        c.n_expert = s.n_expert_r;
        c.n_expert_all = s.n_expert;
        c.expert_base = s.expert_base;
        c.n_ff_exp = s.n_ff_exp_r;
    }

    /* The codes a layer's routed arm reads (MoeFP8::probe's format): int8 and IQR experts read
     * rotated int8, the codebook forms rotated E4M3. */
    static int arm(const MoeFP8::Src& src) {
        const MoeFP8::Config& c = src.cfg;
        if (c.expert_bf16) return kArmBf16;
        return c.expert_i8 || c.expert_iqr ? kArmI8 : kArmE4M3;
    }

    int declare(RadBuilder* b, Names& nm, const Shape& s, const Geom& geom, int l, rad_buf x,
                rad_buf y, const MoeWire& wire, rad_buf dense_gu, rad_buf dense_ff,
                rad_buf dense_out, MoeFP8::Src* src) {
        g = geom;
        layer = l;
        n_embd = s.n_embd;
        moe_layer = s.is_moe[(size_t)l] != 0;
        xf = x;
        yf = y;
        const int64_t T = g.max_tok;
        const auto grp = grp_layer(l);
        const char* mp = nm.ckpt("model.layers.%d.mlp", l);
        auto act = [](rad_buf v) {
            ActFP8 a{};
            a.x = v;
            return a;
        };
        rad_buf sum = 0;
        const char* post_src = nullptr;
        if (!moe_layer) {
            mo = dense_out;
            RAD_ARCH_TRY(mlp.declare(b, nm, g, s, l, nm.f("blk.%d.ffn_gate_up", l),
                                     nm.f("blk.%d.ffn_down", l), mp, s.n_ff_r, xf, mo, dense_gu,
                                     dense_ff));
            sum = mo;
            post_src = nm.ckpt("%s.post_fc2_norm.weight", mp);
        } else {
            w = wire;
            /* The checkpoint names the latent pair by direction in width: routed_expert_down_proj
             * narrows x into the latent in front of the experts, routed_expert_up_proj widens their
             * sum back. */
            RAD_ARCH_TRY(lat_in.declare(b, nm, g, nm.f("blk.%d.ffn_latent_in", l), s.latent,
                                        s.n_embd, RAD_SHARD_NONE, grp,
                                        { nm.ckpt("%s.routed_expert_down_proj.weight", mp) },
                                        act(xf), w.lat.x));
            Geom ge = g;
            ge.n_embd = s.latent;
            ge.n_ff_all = s.n_ff_exp;
            ge.n_ff = s.n_ff_exp_r;
            MoeFP8::Wire mw = w.moe;
            mw.h = w.lat;
            mw.route = xf;
            /* Quantised experts read their codes (MoeWire); bf16 ones the bf16 plane. */
            ge.w_bf16 = src->cfg.expert_bf16 != 0;
            RAD_ARCH_TRY(moe.declare(b, nm, ge, l, mw, *src, false, false));
            RAD_ARCH_TRY(lat_out.declare(b, nm, g, nm.f("blk.%d.ffn_latent_out", l), s.n_embd,
                                         s.latent, RAD_SHARD_NONE, grp,
                                         { nm.ckpt("%s.routed_expert_up_proj.weight", mp) }, w.lat,
                                         w.routed));
            RAD_ARCH_TRY(shared.declare(b, nm, g, s, l, nm.f("blk.%d.ffn_gate_up_shexp", l),
                                        nm.f("blk.%d.ffn_down_shexp", l),
                                        nm.ckpt("%s.shared_experts", mp), s.n_ff_shared_r, xf,
                                        w.sout, w.sgu, w.sff));
            op_join = mk(b, "add",
                         { RAD_RANGE("M", 1, T), RAD_INT("n", s.n_embd), RAD_STR("dtype", "bf16") },
                         {}, { w.routed, w.sout }, { w.routed });
            HY4_NEED(op_join, "add");
            sum = w.routed;
            post_src = nm.ckpt("%s.combine_output_norm.weight", mp);
            RAD_ARCH_TRY(t_logits.declare(b, nm, T, "router_logits", l, RAD_F32, s.n_expert,
                                          w.moe.logits));
            RAD_ARCH_TRY(t_ids.declare(b, nm, T, "router_topk_idx", l, RAD_I32, s.top_k,
                                       w.moe.ids));
            RAD_ARCH_TRY(t_w.declare(b, nm, T, "router_topk_w", l, RAD_F32, s.top_k, w.moe.ew));
            RAD_ARCH_TRY(t_routed_in.declare(b, nm, T, "moe_routed_in", l, RAD_BF16, s.latent,
                                             w.lat.x));
            /* The arms' outputs are a rank's partial sums: tapped at one rank. */
            if (s.world == 1) {
                RAD_ARCH_TRY(t_experts_out.declare(b, nm, T, "moe_experts_out", l, RAD_BF16,
                                                   s.latent, w.lat.x));
                RAD_ARCH_TRY(t_routed_out.declare(b, nm, T, "moe_routed_out", l, RAD_BF16,
                                                  s.n_embd, w.routed));
                RAD_ARCH_TRY(t_shared.declare(b, nm, T, "moe_shared_out", l, RAD_BF16, s.n_embd,
                                              w.sout));
            }
        }
        /* One reduction a block, before the sandwich norm: the dense MLP's partial, or the routed
         * arm's (the rank's experts, widened by the replicated latent_out) plus the shared
         * expert's slice (MoeFP8 runs with no_reduce). */
        {
            int rc = RAD_OK;
            op_ar = decl_all_reduce(b, g, s.n_embd, sum, &rc);
            RAD_ARCH_TRY(rc);
        }
        /* One role, one declared name: the dense layer's sandwich norm is the layer's
         * (`post_fc2_norm`), a MoE layer's the block's (`combine_output_norm`). */
        post_gain = decl_f32(b, nm.f("blk.%d.ffn_post_norm.weight", l), post_src, { s.n_embd },
                             grp);
        if (!post_gain) return RAD_E_INVAL;
        op_post_norm = decl_rmsnorm(b, T, s.n_embd, s.eps, post_gain, sum, yf);
        HY4_NEED(op_post_norm, "rmsnorm");
        RAD_ARCH_TRY(t_out.declare(b, nm, T, "mlp_out", l, RAD_BF16, s.n_embd, yf));
        return RAD_OK;
    }

    bool step(RadCtx* c, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        auto act = [](rad_buf v) {
            ActFP8 a{};
            a.x = v;
            return a;
        };
        rad_buf sum = mo;
        if (!moe_layer) {
            mlp.step(c, xf, mo, T);
        } else {
            lat_in.step(c, act(xf), w.lat.x, T);
            t_routed_in.step(c, brows(w.lat.x, T), T);
            moe.step(c, batch);
            t_logits.step(c, brows(w.moe.logits, T), T);
            t_ids.step(c, brows(w.moe.ids, T), T);
            t_w.step(c, brows(w.moe.ew, T), T);
            t_experts_out.step(c, brows(w.lat.x, T), T);
            lat_out.step(c, w.lat, w.routed, T);
            t_routed_out.step(c, brows(w.routed, T), T);
            shared.step(c, xf, w.sout, T);
            t_shared.step(c, brows(w.sout, T), T);
            RAD_ISSUE_N(c, op_join, T, brows(w.routed, T), brows(w.sout, T), brows(w.routed, T));
            sum = w.routed;
        }
        issue_all_reduce(c, op_ar, sum, T, n_embd);
        RAD_ISSUE_N(c, op_post_norm, T, brows(sum, T), RAD_W(post_gain), brows(yf, T));
        t_out.step(c, brows(yf, T), T);
        return true;
    }
};

}  // namespace hyv4flash
#endif
