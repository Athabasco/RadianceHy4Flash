/* hyv4flash -- Tencent Hy4-flash (`model_type: hy_v4_flash`, HYV4FlashForConditionalGeneration)
 * from its bf16 checkpoint: the text trunk's serving graph at 1, 2 or 4 ranks, and the weight
 * manifest and checkpoint name map rad-convert needs. docs/ARCHITECTURES.md describes the model and its traps.
 *
 *   42 layers over four hyper-connection streams (iHC: a read gate and a write gate a block, no
 *     mixing matrix), hidden 3072, vocab 120832, embedding and lm_head separate
 *   attention: gated MLA (q_lora 1024, kv_lora 512, a shared 64-wide rotary key, a sigmoid gate
 *     of the value width on every head, a learned sink a head, a sandwich norm behind o_proj);
 *     12 sparse layers of 32 heads behind a compressed indexer (16 heads of 128 over keys pooled
 *     four tokens a group) and 30 sliding-window layers of 64 heads (window 512)
 *   feed-forward: layer 0 dense (13824), 41 latent MoE layers: the router and the shared expert
 *     read the hidden state, 512 routed experts at top-16 run in a 1536-wide latent
 *     (routed_expert_down_proj in front, routed_expert_up_proj behind), a sandwich norm closes
 *   Engram: before layers 1, 11, 21 and 31 a hashed bigram/trigram lookup (16 heads of 96 over a
 *     165M-row table a layer, 63.4B parameters in all) gated per stream
 *   DFly2: a 5-layer drafter over the sparse layers' cache and selection
 *   vision: a 27-layer ViT (1152 wide, patch 16) and a two-convolution projector
 *
 * ============================== THE GRAPH (PLAN §1.2) ==============================
 *
 * A step: the token embedding copied into every stream; per layer the Engram add in front of
 * layers 1, 11, 21 and 31 (hyv4flash_engram.h), the attention connection (ihc_pre, attn_norm),
 * the attention (hyv4flash_attn.h, the sparse layers' indexer hyv4flash_indexer.h), ihc_post, the
 * feed-forward connection, the feed-forward (hyv4flash_moe.h), ihc_post; at the end the head's
 * collapse (ihc_pre over hc_head), the final norm and the lm_head into the declared logits.
 *
 * Served: one, two or four ranks (PLAN §1.5: whole heads a rank, the dense and shared widths
 * split, whole experts a rank, the lm_head by vocabulary, two exact all-reduces a layer, the rest
 * replicated; split_ranks, Ffn::place), the text trunk, bf16 weights (the Engram table also as E4M3
 * rows, the bf16 recipe). Refused by name: other worlds, a quantised trunk linear or expert (WP10), a
 * quantised kv_b (DECISIONS U2) or f32 parameter, pictures (WP15). DFly2 and the tower are declared
 * as the checkpoint holds them and not served (WP13, WP15).
 *
 * The caches (PLAN §1.3): the sliding layers' latent rows in a window group (block 16), the sparse
 * layers' in a full group of block 4 (a page is one indexer group, so the indexer's table is the
 * attention's), the indexer's pooled keys (block 4, one rotated key a page) and its raw-key tail
 * (a linear state of ratio + speculation rows), and each Engram layer's token history (a CONV
 * state of ngram - 1 ids).
 *
 * The geometry is read from the checkpoint's config and nothing is assumed about its size, so
 * the convert test and the parity harness run on a model of a few kilobytes. */
#include "hyv4flash_common.h"
#include "hyv4flash_attn.h"
#include "hyv4flash_indexer.h"
#include "hyv4flash_moe.h"
#include "hyv4flash_engram.h"
#include "hyv4flash_calib.h"

#include "chat.h"

#include <memory>
#include <string>
#include <vector>

namespace hyv4flash {

struct Layer {
    bool sliding = false;
    int engram = -1;      /* index into Model::engrams, -1 for none */
    Hyper hc_attn{}, hc_ffn{};
    rad_weight attn_gain = 0, ffn_gain = 0;
    rad_op op_attn_norm = 0, op_ffn_norm = 0;
    Attn attn{};
    Indexer ix{};
    Ffn ffn{};
    Tap t_stream_in{}, t_attn_x{}, t_attn_post{}, t_mid{}, t_ffn_x{}, t_ffn_post{}, t_ffn_in{},
        t_resid{};
};

struct Model {
    Names nm{};
    Shape s{};
    Geom g{};
    int64_t n_weights = 0;
    Trunk tr{};
    AttnWire aw_sparse{}, aw_sliding{};
    IndexWire iw{};
    MoeWire mw{};
    EngramWire ew{};
    rad_buf x0 = 0, dense_gu = 0, dense_ff = 0, dense_out = 0, final_hidden = 0, logit_input = 0,
            logits = 0;
    rad_kvgroup kv_sparse = 0, kv_sliding = 0, kv_index = 0, kv_tail = 0, kv_engram = 0;
    rad_weight token = 0, head = 0, final_gain = 0;
    rad_op embed = 0, op_stream = 0, op_norm = 0, op_gather = 0, op_head = 0;
    Hyper hc_head{};
    std::vector<Layer> layers;
    std::vector<Engram> engrams;
    Tap t_embed{}, t_hc_head{}, t_final{};
    bool force_absorbed = false;
    Calib calib{};   /* --calib-dir: the trunk's Hessians (hyv4flash_calib.h) */
};
static std::unique_ptr<Model> models[MAX_SLOTS];

/* ------------------------------------------------------------------ the config */
static int64_t top(const RadModelMeta* m, const char* key, int64_t dflt) {
    return rad_meta_geti(m, key, dflt);
}
static int64_t txt(const RadModelMeta* m, const char* key, int64_t dflt) {
    return meta_text_geti(m, key, dflt);
}
/* The sliding layers' value where the config states one (`swa_<key>`), else the sparse layers'. */
static int64_t swa(const RadModelMeta* m, const char* key, int64_t sparse) {
    char k[96];
    std::snprintf(k, sizeof k, "swa_%s", key);
    return txt(m, k, sparse);
}

static int read_shape(Shape& s, const RadModelMeta* m) {
    s.n_layer = m->n_layers;
    s.n_embd = m->n_embd;
    s.n_vocab = m->n_vocab;
    s.hc = txt(m, "hc_mult", 0);
    s.q_lora = txt(m, "q_lora_rank", 0);
    s.kv_lora = txt(m, "kv_lora_rank", 0);
    s.sparse.heads = txt(m, "num_attention_heads", 0);
    s.sparse.qk_nope = txt(m, "qk_nope_head_dim", 0);
    s.sparse.qk_rope = txt(m, "qk_rope_head_dim", 0);
    s.sparse.v = txt(m, "v_head_dim", 0);
    s.sliding.heads = swa(m, "num_attention_heads", s.sparse.heads);
    s.sliding.qk_nope = swa(m, "qk_nope_head_dim", s.sparse.qk_nope);
    s.sliding.qk_rope = swa(m, "qk_rope_head_dim", s.sparse.qk_rope);
    s.sliding.v = swa(m, "v_head_dim", s.sparse.v);
    s.n_ff = txt(m, "intermediate_size", 0);
    s.n_expert = m->n_expert;
    s.n_ff_exp = txt(m, "moe_intermediate_size", 0);
    s.latent = txt(m, "routed_expert_hidden_size", 0);
    s.n_ff_shared = s.n_ff_exp * txt(m, "n_shared_experts", 0);
    s.ix_heads = txt(m, "index_n_heads", 0);
    s.ix_dim = txt(m, "index_head_dim", 0);
    s.ix_pool = txt(m, "compressed_indexer_compress_ratio", 0);
    s.ix_group_topk = txt(m, "compressed_indexer_group_topk", 0);
    s.dense_limit = s.ix_group_topk * s.ix_pool;
    s.top_k = txt(m, "num_experts_per_tok", m->n_expert_used);
    s.route_scale = meta_text_getf(m, "routed_scaling_factor", 1.0);
    s.swiglu_limit = meta_text_getf(m, "swiglu_limit", 0.0);
    s.norm_topk = txt(m, "norm_topk_prob", 1) != 0;
    s.eps = m->rms_eps;
    s.hc_eps = meta_text_getf(m, "hc_eps", 1e-6);
    s.hc_mag = meta_text_getf(m, "hc_magnitude", 2.0);
    /* The two layer kinds' rotary bases (HF keys them by layer type; the release config spells
     * the sparse kind `deepseek_sparse_attention`, the port `indexed_attention`). */
    s.theta_sliding = meta_text_getf(m, "rope_parameters.sliding_attention.rope_theta", 1e4);
    s.theta_sparse = meta_text_getf(
        m, "rope_parameters.deepseek_sparse_attention.rope_theta",
        meta_text_getf(m, "rope_parameters.indexed_attention.rope_theta", 1e7));
    /* A sliding row sees `sliding_window` keys, its own included (HF's sliding mask;
     * `window_size` [511, 0] says the same as 511 before it). */
    s.window = txt(m, "sliding_window", 0);
    s.engram_pad = txt(m, "engram_pad_id", -1);
    s.engram_fp8 = txt(m, "engram_fp8_embedding", 0) != 0;
    s.engram_map = txt(m, "engram_lookup_table_size", s.n_vocab);
    s.image_token = top(m, "image_token_id", -1);
    s.video_token = top(m, "video_token_id", -1);

    if (s.n_layer < 1 || s.n_embd < 1 || s.n_vocab < 1 || s.hc < 1 || s.q_lora < 1 ||
        s.kv_lora < 1 || s.sparse.heads < 1 || s.sparse.qk_nope < 1 || s.sparse.qk_rope < 1 ||
        s.sparse.v < 1 || s.sliding.heads < 1 || s.n_ff < 1)
        return refuse("the config lacks one of hidden_size, num_hidden_layers, vocab_size, "
                      "hc_mult, q_lora_rank, kv_lora_rank, num_attention_heads, "
                      "qk_nope_head_dim, qk_rope_head_dim, v_head_dim or intermediate_size");
    /* The graph this plugin maps: MLA with the head-wise gate, sinks and sandwich norms, iHC,
     * and the compressed indexer on the sparse layers. A checkpoint without one of them has
     * other tensors, and mapping it by these names would leave them out without an error. */
    const struct { const char* key; int64_t want; } flags[] = {
        { "use_mla", 1 }, { "gated_mla", 1 }, { "learnable_sink", 1 }, { "sandwich_norm", 1 },
        { "enable_ihc", 1 }, { "use_dsa", 1 }, { "use_compressed_indexer", 1 },
        { "attention_bias", 0 }, { "moe_router_use_sigmoid", 1 },
        { "moe_router_enable_expert_bias", 1 },
    };
    for (const auto& f : flags)
        if (txt(m, f.key, -1) != f.want) {
            fprintf(stderr, "radiance: hyv4flash: the config's %s is %lld; this plugin maps the "
                            "Hy4-flash graph, where it is %lld\n",
                    f.key, (long long)txt(m, f.key, -1), (long long)f.want);
            return RAD_E_UNSUPPORTED;
        }

    RAD_ARCH_TRY(parse_layer_kinds("hyv4flash", meta_text_gets(m, "layer_types", ""),
                                   { "sliding_attention" }, "deepseek_sparse_attention",
                                   &s.is_sliding));
    RAD_ARCH_TRY(parse_layer_kinds("hyv4flash", meta_text_gets(m, "mlp_layer_types", ""),
                                   { "sparse" }, "dense", &s.is_moe));
    if ((int64_t)s.is_sliding.size() != s.n_layer || (int64_t)s.is_moe.size() != s.n_layer)
        return refuse("layer_types and mlp_layer_types must name every layer");
    bool any_sparse = false, any_moe = false;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        any_sparse = any_sparse || !s.is_sliding[(size_t)l];
        any_moe = any_moe || s.is_moe[(size_t)l];
    }
    if (any_sparse && (s.ix_heads < 1 || s.ix_dim < 1 || s.ix_pool < 1 || s.ix_group_topk < 2))
        return refuse("the config has sparse layers and lacks index_n_heads, index_head_dim, "
                      "compressed_indexer_compress_ratio or compressed_indexer_group_topk");
    bool any_sliding = false;
    for (int64_t l = 0; l < s.n_layer; ++l) any_sliding = any_sliding || s.is_sliding[(size_t)l];
    if (any_sliding && s.window < 1) return refuse("the config has sliding layers and no sliding_window");
    if (any_moe && (s.n_expert < 1 || s.n_ff_exp < 1 || s.latent < 1 || s.n_ff_shared < 1))
        return refuse("the config has MoE layers and lacks n_routed_experts, "
                      "moe_intermediate_size, routed_expert_hidden_size or n_shared_experts");

    /* Engram. `engram_layer_ids` are one above the layer the memory sits in front of (the
     * reference builds layer l's from id l + 1), and each id's head moduli are a config row:
     * (max_ngram - 1) orders of n_head_per_ngram primes, whose sum is that layer's table. */
    s.engram_ngram = txt(m, "engram_max_ngram_size", 0);
    const int64_t per = txt(m, "engram_n_head_per_ngram", 0);
    const int64_t embed = txt(m, "engram_n_embed_per_ngram", 0);
    int ids[64];
    const int n_ids = parse_layer_ids(meta_text_gets(m, "engram_layer_ids", ""), s.n_layer + 1,
                                      ids, 64);
    if (n_ids > 0) {
        if (s.engram_ngram < 2 || per < 1 || embed < per || embed % per != 0)
            return refuse("the config has engram_layer_ids and lacks engram_max_ngram_size, "
                          "engram_n_head_per_ngram or an engram_n_embed_per_ngram they divide");
        s.engram_heads = (s.engram_ngram - 1) * per;
        s.engram_head_dim = embed / per;
    }
    for (int i = 0; i < n_ids; ++i) {
        if (ids[i] < 1) return refuse("engram_layer_ids names layer id 0, which has no layer");
        char key[96];
        std::snprintf(key, sizeof key, "engram_layer_head_vocab_sizes.%d", ids[i]);
        int64_t rows = 0, heads = 0;
        for_each_token(meta_text_gets(m, key, ""), [&](const char* t, size_t) {
            rows += std::strtoll(t, nullptr, 10);
            ++heads;
            return true;
        });
        if (heads != s.engram_heads || rows < 1) {
            fprintf(stderr, "radiance: hyv4flash: %s names %lld head moduli; an Engram layer has "
                            "%lld\n", key, (long long)heads, (long long)s.engram_heads);
            return RAD_E_UNSUPPORTED;
        }
        s.engram_layers.push_back(ids[i] - 1);
        s.engram_rows.push_back(rows);
    }

    s.dfly_layers = top(m, "dfly_num_hidden_layers", 0);
    s.v_layers = top(m, "vision_config.num_hidden_layers", 0);
    s.v_embd = top(m, "vision_config.hidden_size", 0);
    s.v_ff = top(m, "vision_config.intermediate_size", 0);
    s.v_patch = top(m, "vision_config.patch_size", 0);
    s.v_chan = top(m, "vision_config.num_channels", 3);
    s.v_merge = top(m, "vision_config.spatial_merge_size", 0);
    s.v_pos = top(m, "vision_config.max_vit_seq_len", 0) + 1;
    return RAD_OK;
}

/* ------------------------------------------------------------------ declaring one weight */
/* The manifest's declarations, for what this build declares and does not serve (DFly2, the tower,
 * quantised routed experts): every weight at the encoding it is stored in, so rad-convert writes the
 * whole release. The served graph's weights are declared by the blocks that read them. */
struct Decl {
    RadBuilder* b;
    Model&      m;
    int         layer;   /* the movement group; -1 for a model-level weight */
    int         access;

    /* The checkpoint's extents of `name` where the model behind the builder has it, else the
     * config's (`dflt`): a builder with no model (a test's) answers nothing. */
    int64_t extent(const char* name, int dim, int64_t dflt) const {
        RadEncoding e{};
        int64_t shape[RAD_MAX_RANK] = {};
        uint32_t rank = 0;
        if (rad_weight_encoding(b, name, &e, shape, &rank) != RAD_OK || (uint32_t)dim >= rank)
            return dflt;
        return shape[dim];
    }
    /* A matrix: bf16 as the checkpoint ships it, or whatever a recipe made of it. */
    int mat(const char* name, std::initializer_list<int64_t> shape) {
        return put(name, RAD_BF16, shape, grp_layer(layer));
    }
    /* A gain, a bias a norm adds, a hyper-connection table: f32, widened from bf16 at load where
     * the checkpoint has it so (the norms and the HC arithmetic run in f32). */
    int f32(const char* name, std::initializer_list<int64_t> shape) {
        return put(name, RAD_F32, shape, grp_layer(layer));
    }
    /* One stored weight, at the encoding the model behind the builder holds it in.
     *
     * A plain weight is one declaration, at `want` where that is the stored float or an exact
     * widening of it, else at the stored dtype. A quantised weight is several planes (codes, a
     * scale or two, a codebook, a zero point), and only a kernel's layout can join planes into one
     * operand; with no op to name that kernel, each plane is declared as its own view, at its own
     * dtype and extents, the codes under the weight's name and the others as `<stem>.<role>`
     * (`blk.3.ffn_down_exps.7.scale` beside `.weight`, the spelling rad_block_moe_fp8.h uses).
     * Every view takes the weight's movement group. The block that will serve the weight declares
     * the views its ops take; these only keep the manifest whole. */
    int put(const char* name, uint32_t want, std::initializer_list<int64_t> shape,
            RadWeightGroup g, int optional = 0) {
        ++m.n_weights;
        RadEncoding e{};
        const bool known = weight_enc(b, name, &e) && e.n_planes > 0;
        if (!known || e.n_planes == 1) {
            uint32_t dt = want;
            if (known) {
                const uint32_t have = e.plane[0].dtype;
                const bool widens = want == RAD_F32 && (have == RAD_BF16 || have == RAD_F16);
                if (!widens) dt = have;
            }
            return decl_w(b, name, dt, shape, access, RAD_SHARD_NONE, g, optional) ? RAD_OK
                                                                                 : RAD_E_INVAL;
        }
        int64_t rows = 1, cols = 1;
        std::vector<int64_t> dims(shape);
        rad_enc_view((uint32_t)dims.size(), dims.data(), &rows, &cols);
        std::string stem(name);
        if (stem.size() > 7 && stem.compare(stem.size() - 7, 7, ".weight") == 0)
            stem.resize(stem.size() - 7);
        for (int k = 0; k < e.n_planes; ++k) {
            const RadEncPlane& pl = e.plane[k];
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&pl, rows, cols, &pr, &pc);
            /* Built from the declared name, so a view carries the scope its weight does. */
            const char* view = k == 0 ? name : m.nm.ckpt("%s.%s", stem.c_str(), pl.role);
            RadWeightDecl d = weight_decl(pl.dtype, { pr, pc }, access, RAD_SHARD_NONE, g, nullptr, 0);
            d.optional = optional;
            d.source = name;
            d.planes = pl.role;
            if (!rad_decl_weight(b, view, &d)) return RAD_E_INVAL;
        }
        return RAD_OK;
    }
};

/* ------------------------------------------------------------------ the drafter's manifest */
/* The drafter's gated MLA, declared as the checkpoint holds it (not served). `scope` is
 * "dfly.blk.<l>", `ap` the checkpoint's `...self_attn`. */
static int attention(Decl& d, Names& nm, const Shape& s, const AttnGeom& a, const char* scope,
                     const char* ap) {
    RadBuilder* b = d.b;
    auto n = [&](const char* leaf) { return nm.f("%s.%s", scope, leaf); };
    const char* q_a = n("attn_q_a.weight");
    const char* q_an = n("attn_q_a_norm.weight");
    const char* q_b = n("attn_q_b.weight");
    const char* kv_a = n("attn_kv_a.weight");
    const char* kv_an = n("attn_kv_a_norm.weight");
    const char* kv_b = n("attn_kv_b.weight");
    const char* gate = n("attn_gate.weight");
    const char* out = n("attn_output.weight");
    const char* post = n("attn_post_norm.weight");
    const char* sinks = n("attn_sinks.weight");
    RAD_ARCH_TRY(map_copy(b, q_a, nm.ckpt("%s.q_a_proj.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, q_an, nm.ckpt("%s.q_a_layernorm.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, q_b, nm.ckpt("%s.q_b_proj.weight", ap)));
    /* kv_a's rows are the latent (kv_lora) then the shared rotary key (qk_rope): one projection,
     * split by the reader, as the checkpoint keeps it (`kv_a_proj_with_mqa`). */
    RAD_ARCH_TRY(map_copy(b, kv_a, nm.ckpt("%s.kv_a_proj_with_mqa.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, kv_an, nm.ckpt("%s.kv_a_layernorm.weight", ap)));
    /* kv_b's rows are a head's key (qk_nope) then its value (v), head after head. */
    RAD_ARCH_TRY(map_copy(b, kv_b, nm.ckpt("%s.kv_b_proj.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, gate, nm.ckpt("%s.linear_gate.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, out, nm.ckpt("%s.o_proj.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, post, nm.ckpt("%s.post_proj_norm.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, sinks, nm.ckpt("%s.learnable_sink_param", ap)));
    RAD_ARCH_TRY(d.mat(q_a, { s.q_lora, s.n_embd }));
    RAD_ARCH_TRY(d.f32(q_an, { s.q_lora }));
    RAD_ARCH_TRY(d.mat(q_b, { a.q_width(), s.q_lora }));
    RAD_ARCH_TRY(d.mat(kv_a, { s.kv_lora + a.qk_rope, s.n_embd }));
    RAD_ARCH_TRY(d.f32(kv_an, { s.kv_lora }));
    RAD_ARCH_TRY(d.mat(kv_b, { a.kv_width(), s.kv_lora }));
    RAD_ARCH_TRY(d.mat(gate, { a.v_width(), s.n_embd }));
    RAD_ARCH_TRY(d.mat(out, { s.n_embd, a.v_width() }));
    RAD_ARCH_TRY(d.f32(post, { s.n_embd }));
    RAD_ARCH_TRY(d.f32(sinks, { a.heads }));
    return RAD_OK;
}

/* A gated MLP: gate and up fused, gate rows first, as every block of the tree reads them.
 * `scope` is the declared stem, `mp` the checkpoint's prefix; `shexp` names the shared expert.
 * A width of 0 is read from the checkpoint's down projection, `ff_dflt` where no model answers. */
static int mlp(Decl& d, Names& nm, const Shape& s, const char* scope, const char* mp, int64_t ff,
               bool shexp, int64_t ff_dflt = 0) {
    RadBuilder* b = d.b;
    const char* gu = nm.f(shexp ? "%s.ffn_gate_up_shexp.weight" : "%s.ffn_gate_up.weight", scope);
    const char* dn = nm.f(shexp ? "%s.ffn_down_shexp.weight" : "%s.ffn_down.weight", scope);
    RAD_ARCH_TRY(map_concat(b, gu, { nm.ckpt("%s.gate_proj.weight", mp),
                                     nm.ckpt("%s.up_proj.weight", mp) }));
    RAD_ARCH_TRY(map_copy(b, dn, nm.ckpt("%s.down_proj.weight", mp)));
    if (ff <= 0) ff = d.extent(dn, 1, ff_dflt);
    RAD_ARCH_TRY(d.mat(gu, { 2 * ff, s.n_embd }));
    RAD_ARCH_TRY(d.mat(dn, { s.n_embd, ff }));
    return RAD_OK;
}

/* DFly2, the release's block drafter (`model.dfly_layers`): `dfly_num_hidden_layers` layers of
 * the sparse layers' attention geometry over the target's cache and selection, each with a dense
 * MLP and no hyper-connection, between a fusion of the target's hidden state with the token
 * embedding (`mtp_fusion`), a hidden-state correction and a confidence head. The transformers
 * port ignores these tensors and their forward is in the serving engines, so this declares them
 * as the checkpoint holds them, fused nowhere but an MLP's gate and up. Its groups follow the
 * trunk's layers. */
static int drafter(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    const char* dp = "model.dfly_layers";
    for (int64_t i = 0; i < s.dfly_layers; ++i) {
        const int grp = (int)(s.n_layer + i);
        Decl d{ b, m, grp, RAD_ACCESS_RARE };
        const char* pre = nm.ckpt("%s.%lld", dp, (long long)i);
        const char* scope = nm.f("dfly.blk.%lld", (long long)i);
        const char* an = nm.f("%s.attn_norm.weight", scope);
        const char* fnorm = nm.f("%s.ffn_norm.weight", scope);
        RAD_ARCH_TRY(map_copy(b, an, nm.ckpt("%s.input_layernorm.weight", pre)));
        RAD_ARCH_TRY(map_copy(b, fnorm, nm.ckpt("%s.post_attention_layernorm.weight", pre)));
        RAD_ARCH_TRY(d.f32(an, { s.n_embd }));
        RAD_ARCH_TRY(attention(d, nm, s, s.sparse, scope, nm.ckpt("%s.self_attn", pre)));
        RAD_ARCH_TRY(d.f32(fnorm, { s.n_embd }));
        /* The config does not state the drafter's MLP width; the checkpoint's down projection
         * does (9216, three times the hidden size, in the release). */
        RAD_ARCH_TRY(mlp(d, nm, s, scope, nm.ckpt("%s.mlp", pre), 0, false, 3 * s.n_embd));
    }
    Decl d{ b, m, (int)(s.n_layer + s.dfly_layers), RAD_ACCESS_RARE };
    const int64_t E = s.n_embd;
    struct { const char* name; const char* src; bool gain; int64_t rows, cols; } one[] = {
        { "dfly.fusion.weight", "mtp_fusion.eh_proj.weight", false, E, 2 * E },
        { "dfly.fusion_embed_norm.weight", "mtp_fusion.enorm.weight", true, E, 0 },
        { "dfly.fusion_hidden_norm.weight", "mtp_fusion.hnorm.weight", true, E, 0 },
        { "dfly.fusion_gate", "mtp_fusion.gate", true, E, 0 },
        { "dfly.correct_embed_norm.weight", "hidden_correction.embed_norm.weight", true, E, 0 },
        { "dfly.correct_hidden_norm.weight", "hidden_correction.hidden_norm.weight", true, E, 0 },
        { "dfly.correct_gate.weight", "hidden_correction.gate_proj.weight", false, E, 2 * E },
        { "dfly.correct_up.weight", "hidden_correction.up_proj.weight", false, E, 2 * E },
        { "dfly.correct_down.weight", "hidden_correction.down_proj.weight", false, E, E },
        { "dfly.output_norm.weight", "final_norm.weight", true, E, 0 },
        { "dfly.confidence.weight", "confidence_head.proj.weight", false, 1, E },
        { "dfly.confidence.bias", "confidence_head.proj.bias", true, 1, 0 },
    };
    for (const auto& w : one) {
        const char* name = nm.f("%s", w.name);
        RAD_ARCH_TRY(map_copy(b, name, nm.ckpt("%s.%s", dp, w.src)));
        if (w.gain) RAD_ARCH_TRY(d.f32(name, { w.rows }));
        else RAD_ARCH_TRY(d.mat(name, { w.rows, w.cols }));
    }
    return RAD_OK;
}

/* The tower (`vit.*`): a patch convolution and learned positions, pre-norm blocks of separate
 * biased q, k, v and o with a GELU MLP, and the projector `perceive` (a norm, a merging
 * convolution, a 1x1 convolution to the hidden size, and the row-end newline embedding). Kept as
 * the checkpoint holds it: this tree's towers each fuse differently, and the fusion belongs to
 * the block that will read it. Its groups follow the drafter's. */
static int tower(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    const int g0 = (int)(s.n_layer + s.dfly_layers + 1);
    const int64_t V = s.v_embd, F = s.v_ff, P = s.v_patch, M = s.v_merge, E = s.n_embd;
    Decl d{ b, m, g0, RAD_ACCESS_PER_REQUEST };
    auto take = [&](const char* name, const char* src, bool flt,
                    std::initializer_list<int64_t> shape) {
        const char* n = nm.f("%s", name);
        RAD_ARCH_TRY(map_copy(b, n, nm.ckpt("%s", src)));
        return flt ? d.f32(n, shape) : d.mat(n, shape);
    };
    RAD_ARCH_TRY(take("v.patch.weight", "vit.embeddings.patch_embedding.weight", false,
                      { V, s.v_chan, P, P }));
    RAD_ARCH_TRY(take("v.patch.bias", "vit.embeddings.patch_embedding.bias", true, { V }));
    const char* pos = nm.f("v.position.weight");
    RAD_ARCH_TRY(map_copy(b, pos, "vit.embeddings.position_embedding.weight"));
    RAD_ARCH_TRY(d.mat(pos, { d.extent(pos, 0, s.v_pos), V }));
    for (int64_t l = 0; l < s.v_layers; ++l) {
        d.layer = g0 + 1 + (int)l;
        const std::string vp = "vit.layers." + std::to_string(l);
        const std::string vn = "v.blk." + std::to_string(l);
        struct { const char* name; const char* src; bool flt; int64_t rows, cols; } w[] = {
            { "attn_norm.weight", "input_layernorm.weight", true, V, 0 },
            { "attn_norm.bias", "input_layernorm.bias", true, V, 0 },
            { "attn_q.weight", "self_attn.q_proj.weight", false, V, V },
            { "attn_q.bias", "self_attn.q_proj.bias", true, V, 0 },
            { "attn_k.weight", "self_attn.k_proj.weight", false, V, V },
            { "attn_k.bias", "self_attn.k_proj.bias", true, V, 0 },
            { "attn_v.weight", "self_attn.v_proj.weight", false, V, V },
            { "attn_v.bias", "self_attn.v_proj.bias", true, V, 0 },
            { "attn_output.weight", "self_attn.o_proj.weight", false, V, V },
            { "attn_output.bias", "self_attn.o_proj.bias", true, V, 0 },
            { "ffn_norm.weight", "post_attention_layernorm.weight", true, V, 0 },
            { "ffn_norm.bias", "post_attention_layernorm.bias", true, V, 0 },
            { "ffn_up.weight", "mlp.dense_h_to_4h.weight", false, F, V },
            { "ffn_up.bias", "mlp.dense_h_to_4h.bias", true, F, 0 },
            { "ffn_down.weight", "mlp.dense_4h_to_h.weight", false, V, F },
            { "ffn_down.bias", "mlp.dense_4h_to_h.bias", true, V, 0 },
        };
        for (const auto& t : w) {
            const char* n = nm.f("%s.%s", vn.c_str(), t.name);
            RAD_ARCH_TRY(map_copy(b, n, nm.ckpt("%s.%s", vp.c_str(), t.src)));
            if (t.flt) RAD_ARCH_TRY(d.f32(n, { t.rows }));
            else RAD_ARCH_TRY(d.mat(n, { t.rows, t.cols }));
        }
    }
    d.layer = g0 + 1 + (int)s.v_layers;
    const int64_t C = V * M * M;
    RAD_ARCH_TRY(take("v.merger.norm.weight", "vit.perceive.norm.weight", true, { V }));
    RAD_ARCH_TRY(take("v.merger.merge.weight", "vit.perceive.proj.0.weight", false,
                      { C, V, M, M }));
    RAD_ARCH_TRY(take("v.merger.merge.bias", "vit.perceive.proj.0.bias", true, { C }));
    RAD_ARCH_TRY(take("v.merger.proj.weight", "vit.perceive.proj.2.weight", false,
                      { E, C, 1, 1 }));
    RAD_ARCH_TRY(take("v.merger.proj.bias", "vit.perceive.proj.2.bias", true, { E }));
    RAD_ARCH_TRY(take("v.merger.newline", "vit.perceive.image_newline", false, { E }));
    return RAD_OK;
}

/* ------------------------------------------------------------------ declare */
/* The KV groups, each declared right before the first op that claims its block size (the first
 * op after a group that carries `block_size` binds it, rad_builder.cpp find_kv_consumer). */
static rad_kvgroup latent_group(RadBuilder* b, Names& nm, bool sliding, int64_t window,
                                int64_t row) {
    RadKVGroupDecl d{};
    d.kind = sliding ? RAD_KV_WINDOW : RAD_KV_FULL;
    d.dtype = RAD_BF16;
    d.window = sliding ? window : 0;
    d.n_head_kv = 1;
    d.head_dim = row / 2;
    return rad_decl_kv_group(b, nm.f(sliding ? "kv_latent_sliding" : "kv_latent_sparse"), &d);
}

/* A layer's routed experts as the manifest declares them, one weight an expert (each its own
 * movement unit of two slots), plane-wise views for a quantised encoding: what rad-convert needs of
 * a layer whose routed arm this build does not serve (Ffn::served). The maps of the experts
 * MoeFP8::probe read are already declared. */
static int expert_manifest(RadBuilder* b, Model& m, int l, const MoeFP8::Config& pc) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    const char* mp = nm.ckpt("model.layers.%d.mlp", l);
    const char* gu_src = nm.ckpt("%s.experts.gate_up_proj", mp);
    const char* dn_src = nm.ckpt("%s.experts.down_proj", mp);
    Decl d{ b, m, l, RAD_ACCESS_CONDITIONAL };
    for (int64_t e = 0; e < s.n_expert; ++e) {
        const char* gu = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, (long long)e);
        const bool probed = pc.probed_layer == l && e >= pc.probed_expert &&
                            e < pc.probed_expert + pc.probed_count;
        if (!probed) RAD_ARCH_TRY(map_slice(b, gu, gu_src, (int)e));
        RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l, (long long)e),
                               dn_src, (int)e));
    }
    for (int64_t e = 0; e < s.n_expert; ++e)
        RAD_ARCH_TRY(d.put(nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, (long long)e), RAD_BF16,
                           { 2 * s.n_ff_exp, s.latent }, grp_expert_slot(l, (int)e, 0)));
    for (int64_t e = 0; e < s.n_expert; ++e)
        RAD_ARCH_TRY(d.put(nm.f("blk.%d.ffn_down_exps.%lld.weight", l, (long long)e), RAD_BF16,
                           { s.latent, s.n_ff_exp }, grp_expert_slot(l, (int)e, 1)));
    rad_note(b, "hyv4flash: layer %d's routed experts are quantised; declared for conversion, not "
                "served by this build (WP10)", l);
    return RAD_OK;
}

static int buffers(RadBuilder* b, Model& m) {
    auto& s = m.s;
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, E = s.n_embd;
    auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
    };
    m.tr.h = bf("streams", { T, s.hc * E });
    m.tr.x = bf("hc_x", { T, E });
    m.tr.post = decl_b(b, nm.f("hc_post"), RAD_F32, { T, s.hc });
    m.tr.xn = bf("block_in", { T, E });
    m.tr.y = bf("block_out", { T, E });
    m.x0 = bf("embed", { T, E });
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, E }, RAD_BUF_PERSIST);
    m.logit_input = bf("logit_input", { g.max_logit_rows, E });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    if (!m.tr.h || !m.tr.x || !m.tr.post || !m.tr.xn || !m.tr.y || !m.x0 || !m.final_hidden ||
        !m.logit_input || !m.logits)
        return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));
    bool any_sparse = false, any_sliding = false, any_moe = false, any_dense = false;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        (s.is_sliding[(size_t)l] ? any_sliding : any_sparse) = true;
        (s.is_moe[(size_t)l] ? any_moe : any_dense) = true;
    }
    if (any_sparse)
        RAD_ARCH_TRY(m.aw_sparse.declare(b, nm, s, s.sparse_r, T, g.max_seqs, "sparse"));
    if (any_sliding)
        RAD_ARCH_TRY(m.aw_sliding.declare(b, nm, s, s.sliding_r, T, g.max_seqs, "sliding"));
    if (any_sparse) RAD_ARCH_TRY(m.iw.declare(b, nm, s, g));
    if (any_moe) RAD_ARCH_TRY(m.mw.declare(b, nm, s, g));
    if (any_dense) {
        m.dense_gu = bf("dense_gate_up", { T, 2 * s.n_ff_r });
        m.dense_ff = bf("dense_ff", { T, s.n_ff_r });
        m.dense_out = bf("dense_out", { T, E });
        if (!m.dense_gu || !m.dense_ff || !m.dense_out) return RAD_E_INVAL;
    }
    if (!s.engram_layers.empty()) RAD_ARCH_TRY(m.ew.declare(b, nm, s, g));
    return RAD_OK;
}

static int trunk_layer(RadBuilder* b, Model& m, int l, MoeFP8::Src* msrc) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    const Geom& g = m.g;
    const int64_t T = g.max_tok, E = s.n_embd;
    Layer& lay = m.layers[(size_t)l];
    lay.sliding = s.is_sliding[(size_t)l] != 0;
    const auto grp = grp_layer(l);
    const char* pre = nm.ckpt("model.layers.%d", l);
    const char* ap = nm.ckpt("%s.self_attn", pre);

    for (size_t i = 0; i < s.engram_layers.size(); ++i)
        if (s.engram_layers[i] == l) {
            lay.engram = (int)i;
            Engram& eg = m.engrams[i];
            RAD_ARCH_TRY(eg.declare(b, nm, s, g, l, s.engram_rows[i], m.ew, m.tr.h, m.kv_engram));
        }
    RAD_ARCH_TRY(lay.t_stream_in.declare(b, nm, T, "stream_in", l, RAD_BF16, s.hc * E, m.tr.h));

    RAD_ARCH_TRY(lay.hc_attn.declare(b, nm, s, T, nm.f("blk.%d.hc_attn", l),
                                     nm.ckpt("%s.hc_attn_layer.hc_pre.hc_fn", pre),
                                     nm.ckpt("%s.hc_attn_layer.hc_pre.hc_base", pre),
                                     nm.ckpt("%s.hc_attn_layer.hc_pre.hc_scale", pre), grp, false,
                                     m.tr.h, m.tr.x, m.tr.post));
    RAD_ARCH_TRY(lay.t_attn_x.declare(b, nm, T, "attn_hc_x", l, RAD_BF16, E, m.tr.x));
    RAD_ARCH_TRY(lay.t_attn_post.declare(b, nm, T, "attn_hc_post", l, RAD_F32, s.hc, m.tr.post));
    lay.attn_gain = decl_f32(b, nm.f("blk.%d.attn_norm.weight", l),
                             nm.ckpt("%s.input_layernorm.weight", pre), { E }, grp);
    if (!lay.attn_gain) return RAD_E_INVAL;
    lay.op_attn_norm = decl_rmsnorm(b, T, E, s.eps, lay.attn_gain, m.tr.x, m.tr.xn);
    HY4_NEED(lay.op_attn_norm, "rmsnorm");

    /* The layer's latent group, declared before its first store; a sparse layer's indexer
     * groups before the first pool. */
    rad_kvgroup& kv = lay.sliding ? m.kv_sliding : m.kv_sparse;
    const bool first_kv = !kv;
    const AttnGeom& a = s.local(lay.sliding);
    if (!kv) kv = latent_group(b, nm, lay.sliding, s.window, s.kv_lora + a.qk_rope);
    if (!kv) return RAD_E_INVAL;
    const int64_t block = lay.sliding ? (int64_t)RAD_KV_BLOCK : s.ix_pool;
    RAD_ARCH_TRY(lay.attn.declare(b, nm, s, g, l, lay.sliding,
                                  lay.sliding ? m.aw_sliding : m.aw_sparse, m.tr.xn, m.tr.y, kv,
                                  block, ap, first_kv));
    if (!lay.sliding) {
        if (!m.kv_tail) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_LINEAR;
            d.dtype = RAD_BF16;
            d.n_head_kv = 1;
            d.state_dim[0] = m.iw.ring;
            d.state_dim[1] = s.ix_dim;
            m.kv_tail = rad_decl_kv_group(b, nm.f("kv_index_tail"), &d);
            if (!m.kv_tail) return RAD_E_INVAL;
        }
        const bool first_ix = !m.kv_index;
        if (!m.kv_index) {
            /* One pooled key of ix_dim a page of four token slots: four rows of ix_dim / 4,
             * K and V halves of ix_dim / 8 each. */
            RadKVGroupDecl d{};
            d.kind = RAD_KV_FULL;
            d.dtype = RAD_BF16;
            d.n_head_kv = 1;
            d.head_dim = s.ix_dim / 8;
            m.kv_index = rad_decl_kv_group(b, nm.f("kv_index_keys"), &d);
            if (!m.kv_index) return RAD_E_INVAL;
            /* The scorer reads every group of the history every step: resident. */
            RAD_ARCH_TRY(rad_kv_group_dense_read(b, m.kv_index));
        }
        RAD_ARCH_TRY(lay.ix.declare(b, nm, s, g, l, m.iw, m.tr.xn, lay.attn.w.qa, m.kv_index,
                                    m.kv_tail, kv, ap, first_ix));
    }
    RAD_ARCH_TRY(lay.attn.declare_core(b, nm, ap, lay.sliding ? 0 : m.iw.sel,
                                       lay.sliding ? 0 : m.iw.len));
    RAD_ARCH_TRY(lay.hc_attn.declare_post(b, s, T, m.tr.h, m.tr.post, m.tr.y));
    RAD_ARCH_TRY(lay.t_mid.declare(b, nm, T, "resid_mid", l, RAD_BF16, s.hc * E, m.tr.h));

    RAD_ARCH_TRY(lay.hc_ffn.declare(b, nm, s, T, nm.f("blk.%d.hc_ffn", l),
                                    nm.ckpt("%s.hc_mlp_layer.hc_pre.hc_fn", pre),
                                    nm.ckpt("%s.hc_mlp_layer.hc_pre.hc_base", pre),
                                    nm.ckpt("%s.hc_mlp_layer.hc_pre.hc_scale", pre), grp, false,
                                    m.tr.h, m.tr.x, m.tr.post));
    RAD_ARCH_TRY(lay.t_ffn_x.declare(b, nm, T, "ffn_hc_x", l, RAD_BF16, E, m.tr.x));
    RAD_ARCH_TRY(lay.t_ffn_post.declare(b, nm, T, "ffn_hc_post", l, RAD_F32, s.hc, m.tr.post));
    lay.ffn_gain = decl_f32(b, nm.f("blk.%d.ffn_norm.weight", l),
                            nm.ckpt("%s.post_attention_layernorm.weight", pre), { E }, grp);
    if (!lay.ffn_gain) return RAD_E_INVAL;
    lay.op_ffn_norm = decl_rmsnorm(b, T, E, s.eps, lay.ffn_gain, m.tr.x, m.tr.xn);
    HY4_NEED(lay.op_ffn_norm, "rmsnorm");
    RAD_ARCH_TRY(lay.t_ffn_in.declare(b, nm, T, "ffn_in_norm", l, RAD_BF16, E, m.tr.xn));
    RAD_ARCH_TRY(lay.ffn.declare(b, nm, s, g, l, m.tr.xn, m.tr.y, m.mw, m.dense_gu, m.dense_ff,
                                 m.dense_out, msrc,
                                 [&](int layer) { return expert_manifest(b, m, layer, msrc->cfg); }));
    RAD_ARCH_TRY(lay.hc_ffn.declare_post(b, s, T, m.tr.h, m.tr.post, m.tr.y));
    RAD_ARCH_TRY(lay.t_resid.declare(b, nm, T, "resid", l, RAD_BF16, s.hc * E, m.tr.h));
    RAD_ARCH_TRY(m.calib.layer(b, g, s, l, m.tr.xn, lay.attn, !lay.sliding, lay.ffn,
                               lay.engram >= 0 ? &m.engrams[(size_t)lay.engram] : nullptr));
    return RAD_OK;
}

/* This rank's share of the model (PLAN §1.5, Shape's comment): whole heads a rank on both
 * attention kinds, the dense MLP's and the shared expert's widths evenly. 1, 2 or 4 ranks. The
 * routed experts' placement follows their format (Ffn::place). */
static int split_ranks(Shape& s, int world, int rank) {
    s.world = world;
    s.rank = rank;
    if (world != 1 && world != 2 && world != 4) {
        fprintf(stderr, "radiance: hyv4flash: --tp %d; the head and width splits are built for 1, "
                        "2 or 4 ranks\n", world);
        return RAD_E_UNSUPPORTED;
    }
    s.sparse_r = s.sparse;
    s.sliding_r = s.sliding;
    s.n_ff_r = s.n_ff;
    s.n_ff_shared_r = s.n_ff_shared;
    s.n_expert_r = s.n_expert;
    s.n_ff_exp_r = s.n_ff_exp;
    bool any_sparse = false, any_sliding = false, any_moe = false, any_dense = false;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        (s.is_sliding[(size_t)l] ? any_sliding : any_sparse) = true;
        (s.is_moe[(size_t)l] ? any_moe : any_dense) = true;
    }
    const struct { bool on; const char* what; int64_t n; } parts[] = {
        { any_sparse, "the sparse layers' attention heads", s.sparse.heads },
        { any_sliding, "the sliding layers' attention heads", s.sliding.heads },
        { any_dense, "the dense MLP's intermediate_size", s.n_ff },
        { any_moe, "the shared expert's width", s.n_ff_shared },
        { true, "the vocabulary (lm_head)", s.n_vocab },
    };
    for (const auto& p : parts)
        if (p.on && p.n % world) {
            fprintf(stderr, "radiance: hyv4flash: %s (%lld) do not split evenly over %d ranks\n",
                    p.what, (long long)p.n, world);
            return RAD_E_UNSUPPORTED;
        }
    s.sparse_r.heads = s.sparse.heads / world;
    s.sliding_r.heads = s.sliding.heads / world;
    s.n_ff_r = s.n_ff / world;
    s.n_ff_shared_r = s.n_ff_shared / world;
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS ||
        rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->kv_dtype != RAD_DT_INVALID && ctx->kv_dtype != RAD_BF16)
        return refuse("the latent and indexer caches are bf16; run with --kv-cache-dtype bf16");
    auto& slot = models[rad_build_slot(ctx)];
    slot = std::make_unique<Model>();
    Model& m = *slot;
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    Shape& s = m.s;
    Geom& g = m.g;
    RAD_ARCH_TRY(map_cli_to_env(ctx));   /* before the first block reads RADIANCE_CALIB_* */
    RAD_ARCH_TRY(read_shape(s, meta));
    RAD_ARCH_TRY(split_ranks(s, ctx->world_size ? ctx->world_size : 1, ctx->rank));
    RAD_ARCH_TRY(geom_from(g, meta, ctx, "bf16", RAD_BF16));
    /* Every linear of this build reads the bf16 plane (a quantised one is refused by its
     * LinearFP8): no activation codes are declared. */
    g.w_bf16 = true;
    {
        const char* e = std::getenv("RADIANCE_HY4_PREFILL");
        m.force_absorbed = e && !std::strcmp(e, "absorbed");
    }

    /* The token embedding and the head. The release ships its own lm_head (`tie_word_embeddings`
     * false); a tied fine-tune reads the embedding. The reference runs the head in f32: the
     * logits are f32 accumulations over the bf16 rows. */
    const char* token = nm.f("token_embd.weight");
    const char* headn = nm.f("output.weight");
    RAD_ARCH_TRY(map_copy(b, token, "model.embed_tokens.weight"));
    RAD_ARCH_TRY(map_copy_alt(b, headn, "lm_head.weight", "model.embed_tokens.weight"));
    for (const char* n : { token, headn })
        if (!is_plain_bf16(b, n))
            return refuse("a quantised embedding or lm_head is not served yet (WP10)");
    /* The embedding is replicated (no reduction behind the lookup); the head is split by
     * vocabulary, each rank's logits [rows, n_vocab / world] (Geom::n_vocab, vocab_off). */
    m.token = decl_w(b, token, RAD_BF16, { s.n_vocab, s.n_embd }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                     grp_model());
    m.head = decl_w(b, headn, RAD_BF16, { g.n_vocab, s.n_embd }, RAD_ACCESS_VOCAB, RAD_SHARD_ROW,
                    grp_model());
    m.final_gain = decl_f32(b, nm.f("output_norm.weight"), "model.norm.weight", { s.n_embd },
                            grp_model());
    if (!m.token || !m.head || !m.final_gain) return RAD_E_INVAL;
    m.n_weights += 3;

    /* The routed arms' format, before any buffer (MoeFP8::probe maps each probed expert). */
    std::vector<MoeFP8::Src> msrc((size_t)s.n_layer);
    bool any_quant = false, calib = false;
    for (int l = 0; l < (int)s.n_layer; ++l)
        if (s.is_moe[(size_t)l]) {
            RAD_ARCH_TRY(Ffn::probe(b, nm, s, l, &msrc[(size_t)l]));
            any_quant = any_quant || !Ffn::served(msrc[(size_t)l]);
            calib = calib || msrc[(size_t)l].cfg.calib;
        }
    /* Quantised experts are declared whole for rad-convert and refused at the step (Ffn::served);
     * at more ranks that would load every expert on every rank first, so it is refused here. */
    if (any_quant && s.world > 1)
        return refuse("quantised routed experts are not served yet (WP10); at more than one rank "
                      "they would be declared whole on every rank");
    RAD_ARCH_TRY(Ffn::place(s, calib || !any_quant));
    for (int l = 0; l < (int)s.n_layer; ++l)
        if (s.is_moe[(size_t)l] && Ffn::served(msrc[(size_t)l]))
            Ffn::apply(s, msrc[(size_t)l].cfg);

    RAD_ARCH_TRY(buffers(b, m));
    const int64_t T = g.max_tok, E = s.n_embd;
    m.embed = mk(b, "embed_lookup",
                 { RAD_RANGE("M", 1, T), RAD_INT("n_embd", E), RAD_INT("n_vocab", s.n_vocab),
                   RAD_STR("dtype", "bf16") },
                 { m.token }, {}, { m.x0 });
    m.op_stream = decl_copy(b, T, E, RAD_BF16, m.x0, m.tr.h);
    HY4_NEED(m.embed, "embed_lookup");
    HY4_NEED(m.op_stream, "cast");
    RAD_ARCH_TRY(m.t_embed.declare(b, nm, T, "embed", -1, RAD_BF16, E, m.x0));

    if (!s.engram_layers.empty()) {
        /* One CONV state a layer (each Engram layer advances its own window once a step): the last
         * ngram - 1 token ids, i32. */
        RadKVGroupDecl d{};
        d.kind = RAD_KV_CONV;
        d.dtype = RAD_I32;
        d.conv_width = s.engram_ngram;
        d.n_head_kv = 1;
        d.head_dim = 1;
        m.kv_engram = rad_decl_kv_group(b, nm.f("kv_engram_tokens"), &d);
        if (!m.kv_engram) return RAD_E_INVAL;
        if (s.engram_pad < 0) return refuse("the config has Engram layers and no engram_pad_id");
    }
    RAD_ARCH_TRY(m.calib.begin(b, nm, g, ctx, s));
    m.engrams.resize(s.engram_layers.size());
    m.layers.resize((size_t)s.n_layer);
    for (int l = 0; l < (int)s.n_layer; ++l) RAD_ARCH_TRY(trunk_layer(b, m, l, &msrc[(size_t)l]));

    /* The head: the streams' collapse, the final norm, the lm_head over the sampled rows. */
    RAD_ARCH_TRY(m.hc_head.declare(b, nm, s, T, nm.f("hc_head"), "model.hc_head.hc_head_fn",
                                   "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale",
                                   grp_model(), true, m.tr.h, m.tr.x, 0));
    RAD_ARCH_TRY(m.t_hc_head.declare(b, nm, T, "hc_head_out", -1, RAD_BF16, E, m.tr.x));
    m.op_norm = decl_rmsnorm(b, T, E, s.eps, m.final_gain, m.tr.x, m.final_hidden);
    RAD_ARCH_TRY(m.t_final.declare(b, nm, T, "final_hidden", -1, RAD_BF16, E, m.final_hidden));
    m.op_gather = mk(b, "gather_rows",
                     { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", E),
                       RAD_STR("dtype", "bf16") },
                     {}, { m.final_hidden }, { m.logit_input });
    m.op_head = mk(b, "logits_gemm",
                   { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                     RAD_INT("n_embd", E), RAD_STR("dtype", "bf16") },
                   { m.head }, { m.logit_input }, { m.logits });
    HY4_NEED(m.op_norm && m.op_gather && m.op_head, "rmsnorm / gather_rows / logits_gemm");
    RAD_ARCH_TRY(m.calib.output(b, g, s, m.final_hidden));

    /* A container carries what the model can serve, so the drafter and the tower are declared
     * wherever the checkpoint has them and the caller left room (rad-convert's --max-spec 0
     * leaves the drafter out). Neither is served by this build. */
    const bool dfly = s.dfly_layers > 0 && ctx->max_spec > 0;
    const bool vit = s.v_layers > 0 && s.v_embd > 0 && s.v_patch > 0 && s.v_merge > 0 &&
                     ctx->max_enc_patches > 0;
    if (dfly) RAD_ARCH_TRY(drafter(b, m));
    if (vit) RAD_ARCH_TRY(tower(b, m));

    int64_t n_sparse = 0, n_moe = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        n_sparse += !s.is_sliding[(size_t)l];
        n_moe += s.is_moe[(size_t)l] != 0;
    }
    rad_note(b,
             "Hy4-flash (hyv4flash): %lld layers over %lld streams, %lld sparse + %lld sliding "
             "MLA (window %lld, dense to %lld keys), %lld latent MoE of %lld experts top-%lld, "
             "%zu Engram tables (eps %g), DFly2 %s, vision %s; TP%d (rank %d: %lld/%lld sparse "
             "and %lld/%lld sliding heads, experts %s), bf16%s",
             (long long)s.n_layer, (long long)s.hc, (long long)n_sparse,
             (long long)(s.n_layer - n_sparse), (long long)s.window, (long long)s.dense_limit,
             (long long)n_moe, (long long)s.n_expert, (long long)s.top_k, s.engram_layers.size(),
             s.engram_layers.empty() ? 0.0 : engram_eps(),
             dfly ? "declared (not served)" : "left out", vit ? "declared (not served)" : "left out",
             s.world, s.rank, (long long)s.sparse_r.heads, (long long)s.sparse.heads,
             (long long)s.sliding_r.heads, (long long)s.sliding.heads,
             s.world == 1 ? "all" : s.experts_whole ? "whole per rank" : "sliced per rank",
             taps_on() ? ", debug taps" : "");
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
static StepPlan plan(const Model& m, const RadBatch* batch) {
    StepPlan p;
    p.T = batch->n_tok;
    p.S = batch->n_seq;
    batch_split(batch, &p.D, &p.DT);
    p.P = p.S - p.D;
    /* The longest context after the step, from the host mirrors. */
    int64_t ctx_max = (int64_t)batch->max_ctx_len + batch->max_q_len;
    int64_t zeros = 0;
    if (batch->positions_h) {
        ctx_max = 0;
        for (int64_t t = 0; t < p.T; ++t) {
            ctx_max = std::max<int64_t>(ctx_max, (int64_t)batch->positions_h[t] + 1);
            if (t >= p.DT && batch->positions_h[t] == 0) ++zeros;
        }
    }
    p.select = m.s.dense_limit > 0 && ctx_max > m.s.dense_limit;
    /* Every prefill sequence of the step starts at position 0 (one row at 0 each): its keys are
     * this step's rows, which the non-absorbed path reads. */
    p.nonabs = !m.force_absorbed && batch->positions_h && p.P > 0 && zeros == p.P;
    /* The indexer's live block columns: the reach the batch bounds on the host. */
    const int64_t r = m.s.ix_pool > 0 ? m.s.ix_pool : 4;
    int64_t live = ((int64_t)batch->max_ctx_len + p.T + batch->n_spec + r) / r + 1;
    if (live > m.iw.blocks) live = m.iw.blocks;
    if (m.kv_index)
        if (const auto* ib = kv_batch(batch, m.kv_index))
            if (live > ib->block_table_pitch) live = ib->block_table_pitch;
    p.live = live < 1 ? 1 : live;
    return p;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = *models[rad_slot(c)];
    const auto& s = m.s;
    const int64_t T = batch->n_tok, E = s.n_embd;
    if (batch->enc || batch->draft_pass || batch->gen) {
        rad_step_fail(c, "hyv4flash serves the text trunk; pictures, the DFly2 drafter and "
                         "generator passes are not built (WP13, WP15)");
        return;
    }
    if (batch->n_mm_rows > 0) {
        rad_step_fail(c, "hyv4flash serves text; this step carries media rows (WP15)");
        return;
    }
    const StepPlan p = plan(m, batch);
    RAD_ISSUE_N(c, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.x0, T));
    m.t_embed.step(c, brows(m.x0, T), T);
    for (int64_t j = 0; j < s.hc; ++j)
        RAD_ISSUE_N(c, m.op_stream, T, brows(m.x0, T), bcol(m.tr.h, j * E, E, T));

    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer& l = m.layers[i];
        if (l.engram >= 0) {
            if (!m.engrams[(size_t)l.engram].step(c, batch)) return;
            m.calib.after_engram(c, (int)i, T);
        }
        l.t_stream_in.step(c, brows(m.tr.h, T), T);
        l.hc_attn.pre(c, m.tr.h, m.tr.x, m.tr.post, T);
        l.t_attn_x.step(c, brows(m.tr.x, T), T);
        l.t_attn_post.step(c, brows(m.tr.post, T), T);
        RAD_ISSUE_N(c, l.op_attn_norm, T, brows(m.tr.x, T), RAD_W(l.attn_gain), brows(m.tr.xn, T));
        if (!l.attn.front(c, batch)) return;
        m.calib.after_front(c, (int)i, l.attn, T);
        if (!l.sliding && !l.ix.step(c, batch, p)) return;
        l.attn.core(c, batch, p, m.iw.sel, m.iw.len);
        m.calib.after_attn(c, (int)i, T);
        l.hc_attn.post_add(c, m.tr.y, m.tr.post, m.tr.h, T);
        l.t_mid.step(c, brows(m.tr.h, T), T);

        l.hc_ffn.pre(c, m.tr.h, m.tr.x, m.tr.post, T);
        l.t_ffn_x.step(c, brows(m.tr.x, T), T);
        l.t_ffn_post.step(c, brows(m.tr.post, T), T);
        RAD_ISSUE_N(c, l.op_ffn_norm, T, brows(m.tr.x, T), RAD_W(l.ffn_gain), brows(m.tr.xn, T));
        l.t_ffn_in.step(c, brows(m.tr.xn, T), T);
        m.calib.after_ffn_norm(c, (int)i, T);
        if (!l.ffn.step(c, batch)) return;
        m.calib.after_ffn(c, (int)i, T);
        l.hc_ffn.post_add(c, m.tr.y, m.tr.post, m.tr.h, T);
        l.t_resid.step(c, brows(m.tr.h, T), T);
    }
    m.hc_head.pre(c, m.tr.h, m.tr.x, 0, T);
    m.t_hc_head.step(c, brows(m.tr.x, T), T);
    RAD_ISSUE_N(c, m.op_norm, T, brows(m.tr.x, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    m.t_final.step(c, brows(m.final_hidden, T), T);
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(c, m.op_head, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
    m.calib.after_step(c, T);
}
} // namespace hyv4flash

#ifndef RAD_ARCH_NO_EXPORTS
/* The model's state is indexed by slot (rad_build_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(hyv4flash, "hyv4flash", "", "0.2.0",
                "Tencent Hy4-flash: the text trunk at 1, 2 or 4 ranks (bf16), the weight manifest and "
                "checkpoint name map")
RAD_ARCH_CHAT_FORMAT(hyv4flash)
#endif
