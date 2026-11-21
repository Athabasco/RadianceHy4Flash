/* hyv4flash -- Tencent Hy4-flash (`model_type: hy_v4_flash`, HYV4FlashForConditionalGeneration)
 * from its bf16 checkpoint: the weight manifest and the checkpoint name map, which is what
 * rad-convert needs to quantise the release into a container. docs/ARCHITECTURES.md describes the
 * model and its traps.
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
 * WHAT THIS PLUGIN IS. It declares every weight the release ships, under the logical names the
 * tree's blocks use for the same roles (blk.N.attn_q_a, blk.N.ffn_gate_up_exps.E, ...), each at
 * the encoding the container, checkpoint or recipe holds it in, and maps each to its checkpoint
 * tensors: the stacked experts one weight an expert, gate and up fused, the Engram's per-stream
 * projections and gains joined. It declares no op: this build has no kernel for the compressed
 * indexer's pooling, the Engram gate or DFly2, and a graph that stopped short of them would look
 * servable. rad_arch_step therefore fails by name, and a recipe's encodings are checked by the
 * quantisers and the core's decoder only, not by a kernel's layout hook (docs/ARCHITECTURES.md
 * lists which encodings the tree's kernels read for the same roles in other models).
 *
 * The geometry is read from the checkpoint's config and nothing is assumed about its size, so
 * the convert test runs on a model of a few kilobytes. */
#include <arch/rad_arch.h>

#include "chat.h"

#include <memory>
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
};

struct Shape {
    int64_t n_layer = 0, n_embd = 0, n_vocab = 0, hc = 0;
    int64_t q_lora = 0, kv_lora = 0;
    AttnGeom sparse{}, sliding{};
    std::vector<int> is_sliding;   /* per layer */
    std::vector<int> is_moe;       /* per layer */
    /* feed-forward */
    int64_t n_ff = 0, n_expert = 0, n_ff_exp = 0, latent = 0, n_ff_shared = 0;
    /* the compressed indexer */
    int64_t ix_heads = 0, ix_dim = 0, ix_pool = 0;
    /* Engram: `layers` are layer indices (the config's ids are one above them) */
    std::vector<int> engram_layers;
    std::vector<int64_t> engram_rows;   /* each layer's table, the sum of its head moduli */
    int64_t engram_ngram = 0, engram_heads = 0, engram_head_dim = 0, engram_map = 0;
    /* DFly2 and the tower */
    int64_t dfly_layers = 0;
    int64_t v_layers = 0, v_embd = 0, v_ff = 0, v_patch = 0, v_chan = 0, v_merge = 0, v_pos = 0;
};

struct Model {
    Names nm{};
    Shape s{};
    int64_t n_weights = 0;
};
static std::unique_ptr<Model> models[MAX_SLOTS];

static int refuse(const char* what) {
    fprintf(stderr, "radiance: hyv4flash: %s\n", what);
    return RAD_E_UNSUPPORTED;
}

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
    if (any_sparse && (s.ix_heads < 1 || s.ix_dim < 1 || s.ix_pool < 1))
        return refuse("the config has sparse layers and lacks index_n_heads, index_head_dim or "
                      "compressed_indexer_compress_ratio");
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

/* ------------------------------------------------------------------ the blocks */
/* Gated MLA. `scope` is "blk.<l>" or "dfly.blk.<l>", `ap` the checkpoint's `...self_attn`. */
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

/* The compressed indexer of a sparse layer. Its pooling (two projections, the in-group position
 * bias and the pooled key's norm) is f32 in the checkpoint and in the reference, which keeps it
 * there by name: a softmax over four slots loses near-ties in bf16 and recalls other groups. */
static int indexer(Decl& d, Names& nm, const Shape& s, int l, const char* ap) {
    RadBuilder* b = d.b;
    const char* ip = nm.ckpt("%s.compressed_indexer", ap);
    const char* query = nm.f("blk.%d.index_query.weight", l);
    const char* head = nm.f("blk.%d.index_head.weight", l);
    const char* key = nm.f("blk.%d.index_key.weight", l);
    const char* key_n = nm.f("blk.%d.index_key_norm.weight", l);
    const char* pool_v = nm.f("blk.%d.index_pool_value.weight", l);
    const char* pool_g = nm.f("blk.%d.index_pool_gate.weight", l);
    const char* pool_n = nm.f("blk.%d.index_pool_norm.weight", l);
    const char* ape = nm.f("blk.%d.index_ape", l);
    RAD_ARCH_TRY(map_copy(b, query, nm.ckpt("%s.query_projection.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, head, nm.ckpt("%s.head_weight_projection.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, key, nm.ckpt("%s.token_key_projection.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, key_n, nm.ckpt("%s.token_key_norm.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, pool_v, nm.ckpt("%s.key_value_projection.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, pool_g, nm.ckpt("%s.key_gate_projection.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, pool_n, nm.ckpt("%s.key_norm.weight", ip)));
    RAD_ARCH_TRY(map_copy(b, ape, nm.ckpt("%s.intra_group_ape", ip)));
    RAD_ARCH_TRY(d.mat(query, { s.ix_heads * s.ix_dim, s.q_lora }));
    RAD_ARCH_TRY(d.mat(head, { s.ix_heads, s.n_embd }));
    RAD_ARCH_TRY(d.mat(key, { s.ix_dim, s.n_embd }));
    RAD_ARCH_TRY(d.f32(key_n, { s.ix_dim }));
    RAD_ARCH_TRY(d.f32(pool_v, { s.ix_dim, s.ix_dim }));
    RAD_ARCH_TRY(d.f32(pool_g, { s.ix_dim, s.ix_dim }));
    RAD_ARCH_TRY(d.f32(pool_n, { s.ix_dim }));
    RAD_ARCH_TRY(d.f32(ape, { s.ix_pool, s.ix_dim }));
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

/* A latent MoE layer. The experts are two stacked tensors in the checkpoint, gate_up
 * [n_expert, 2 * n_ff_exp, latent] (gate rows first) and down [n_expert, latent, n_ff_exp], and
 * one weight an expert here, each expert its own movement unit of two slots: the unit the
 * placement planner tiers and the heat engine promotes (rad_block_moe_fp8.h). */
static int moe(Decl& d, Names& nm, const Shape& s, int l, const char* mp) {
    RadBuilder* b = d.b;
    const char* router = nm.f("blk.%d.ffn_gate_inp.weight", l);
    const char* corr = nm.f("blk.%d.ffn_gate_inp.correction", l);
    const char* lat_in = nm.f("blk.%d.ffn_latent_in.weight", l);
    const char* lat_out = nm.f("blk.%d.ffn_latent_out.weight", l);
    const char* post = nm.f("blk.%d.ffn_post_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, router, nm.ckpt("%s.gate.weight", mp)));
    RAD_ARCH_TRY(map_copy(b, corr, nm.ckpt("%s.gate.e_score_correction_bias", mp)));
    /* The checkpoint names the pair by direction in width: `routed_expert_down_proj` narrows the
     * hidden state into the latent in front of the experts, `routed_expert_up_proj` widens their
     * sum back. Neither is an expert's own down or up projection. */
    RAD_ARCH_TRY(map_copy(b, lat_in, nm.ckpt("%s.routed_expert_down_proj.weight", mp)));
    RAD_ARCH_TRY(map_copy(b, lat_out, nm.ckpt("%s.routed_expert_up_proj.weight", mp)));
    RAD_ARCH_TRY(map_copy(b, post, nm.ckpt("%s.combine_output_norm.weight", mp)));
    /* The router scores in f32 (the reference casts both operands), so it stays a float. */
    RAD_ARCH_TRY(d.mat(router, { s.n_expert, s.n_embd }));
    RAD_ARCH_TRY(d.f32(corr, { s.n_expert }));
    RAD_ARCH_TRY(d.mat(lat_in, { s.latent, s.n_embd }));
    RAD_ARCH_TRY(d.mat(lat_out, { s.n_embd, s.latent }));
    RAD_ARCH_TRY(mlp(d, nm, s, nm.f("blk.%d", l), nm.ckpt("%s.shared_experts", mp),
                     s.n_ff_shared, true));
    RAD_ARCH_TRY(d.f32(post, { s.n_embd }));

    const char* gu_src = nm.ckpt("%s.experts.gate_up_proj", mp);
    const char* dn_src = nm.ckpt("%s.experts.down_proj", mp);
    const int access = d.access;
    d.access = RAD_ACCESS_CONDITIONAL;
    /* Every map of the layer before its first declaration: an encoding is looked up through its
     * map. */
    for (int64_t e = 0; e < s.n_expert; ++e) {
        RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, (long long)e),
                               gu_src, (int)e));
        RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l, (long long)e),
                               dn_src, (int)e));
    }
    for (int64_t e = 0; e < s.n_expert; ++e) {
        const char* w = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, (long long)e);
        RAD_ARCH_TRY(d.put(w, RAD_BF16, { 2 * s.n_ff_exp, s.latent },
                           grp_expert_slot(l, (int)e, 0)));
    }
    for (int64_t e = 0; e < s.n_expert; ++e) {
        const char* w = nm.f("blk.%d.ffn_down_exps.%lld.weight", l, (long long)e);
        RAD_ARCH_TRY(d.put(w, RAD_BF16, { s.latent, s.n_ff_exp }, grp_expert_slot(l, (int)e, 1)));
    }
    d.access = access;
    return RAD_OK;
}

/* Joins `n` per-stream (or per-head) checkpoint tensors `<prefix><i><suffix>` along dim 0, eight
 * a map: successive CONCAT maps for one name extend it in call order. */
static int map_join(RadBuilder* b, Names& nm, const char* declared, const char* prefix,
                    const char* suffix, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
        for (int64_t k = i; k < n && k < i + 8; ++k)
            m.src[m.n_src++] = nm.ckpt("%s%lld%s", prefix, (long long)k, suffix);
        RAD_ARCH_TRY(rad_decl_name_map(b, &m));
    }
    return RAD_OK;
}

/* The Engram memory in front of layer l. The reference keeps one module a stream (four key
 * projections, four key and four query norms, four branch scales) and one norm a hash head;
 * each family is joined here along dim 0, stream 0 or head 0 first, so the four key projections
 * are one GEMM and each gain family one operand.
 *
 * `engram_fp8_embedding` makes every looked-up row an E4M3 round trip under its own absmax
 * scale, so the bf16 table is not what the model reads. A recipe may store the table as E4M3 rows
 * (data/recipes/hy4-flash-bf16.recipe), at half the bytes and within one code step of the
 * reference, whose own division is in bf16 (the recipe's header has the measurement). Stored
 * bf16, the round trip is the reader's to apply.
 *
 * The hash is not in the checkpoint: the reference draws each layer's odd multipliers from
 * NumPy's generator at load and reads the head moduli from the config. They are declared
 * optional, under the names data/hy4flash_engram_tables.py writes for `rad-convert --add`. */
static int engram(Decl& d, Names& nm, const Shape& s, int l, int64_t rows, const char* pre) {
    RadBuilder* b = d.b;
    const char* ep = nm.ckpt("%s.engram_layer", pre);
    const int64_t E = s.n_embd, HD = s.engram_head_dim, H = s.engram_heads, W = H * HD;
    const char* table = nm.f("blk.%d.engram_table.weight", l);
    const char* map = nm.f("blk.%d.engram_token_map", l);
    const char* mult = nm.f("blk.%d.engram_mult", l);
    const char* vsz = nm.f("blk.%d.engram_vocab_sizes", l);
    const char* off = nm.f("blk.%d.engram_offsets", l);
    const char* head_n = nm.f("blk.%d.engram_head_norm.weight", l);
    const char* key = nm.f("blk.%d.engram_key.weight", l);
    const char* key_b = nm.f("blk.%d.engram_key.bias", l);
    const char* key_n = nm.f("blk.%d.engram_key_norm.weight", l);
    const char* query_n = nm.f("blk.%d.engram_query_norm.weight", l);
    const char* value = nm.f("blk.%d.engram_value.weight", l);
    const char* value_b = nm.f("blk.%d.engram_value.bias", l);
    const char* value_g = nm.f("blk.%d.engram_value_gain.weight", l);
    const char* branch = nm.f("blk.%d.engram_branch_scale.weight", l);
    RAD_ARCH_TRY(map_copy(b, table, nm.ckpt("%s.multi_head_embedding.embedding.weight", ep)));
    RAD_ARCH_TRY(map_copy(b, map, nm.ckpt("%s.lookup_table", ep)));
    RAD_ARCH_TRY(map_copy(b, mult, nm.ckpt("engram.%d.hash_multipliers", l)));
    RAD_ARCH_TRY(map_copy(b, vsz, nm.ckpt("engram.%d.head_vocab_sizes", l)));
    RAD_ARCH_TRY(map_copy(b, off, nm.ckpt("engram.%d.head_offsets", l)));
    RAD_ARCH_TRY(map_join(b, nm, head_n, nm.ckpt("%s.lookup_per_head_norms.", ep), ".weight", H));
    RAD_ARCH_TRY(map_join(b, nm, key, nm.ckpt("%s.key_projs.", ep), ".weight", s.hc));
    RAD_ARCH_TRY(map_join(b, nm, key_b, nm.ckpt("%s.key_projs.", ep), ".bias", s.hc));
    RAD_ARCH_TRY(map_join(b, nm, key_n, nm.ckpt("%s.norm_key.", ep), ".weight", s.hc));
    RAD_ARCH_TRY(map_join(b, nm, query_n, nm.ckpt("%s.norm_query.", ep), ".weight", s.hc));
    RAD_ARCH_TRY(map_copy(b, value, nm.ckpt("%s.value_proj.weight", ep)));
    RAD_ARCH_TRY(map_copy(b, value_b, nm.ckpt("%s.value_proj.bias", ep)));
    RAD_ARCH_TRY(map_copy(b, value_g, nm.ckpt("%s.unified_pre_gamma", ep)));
    RAD_ARCH_TRY(map_join(b, nm, branch, nm.ckpt("%s.unified_per_hc_scales.", ep), "", s.hc));

    /* A vocabulary table, though not the token one: read a few rows a token, far too large for a
     * card (rad_block_ple.h places Flash-Next's the same way). */
    const int access = d.access;
    d.access = RAD_ACCESS_VOCAB;
    RAD_ARCH_TRY(d.put(table, RAD_BF16, { rows, HD }, grp_layer(l)));
    d.access = access;
    /* The token map's length is the tokenizer's, below the embedding's padded vocabulary. */
    RAD_ARCH_TRY(d.put(map, RAD_I32, { d.extent(map, 0, s.n_vocab) }, grp_layer(l)));
    /* I64: a multiplier times a compressed token id fills 63 bits. */
    RAD_ARCH_TRY(d.put(mult, RAD_I64, { s.engram_ngram }, grp_layer(l), 1));
    RAD_ARCH_TRY(d.put(vsz, RAD_I64, { H }, grp_layer(l), 1));
    RAD_ARCH_TRY(d.put(off, RAD_I64, { H }, grp_layer(l), 1));
    RAD_ARCH_TRY(d.f32(head_n, { W }));
    RAD_ARCH_TRY(d.mat(key, { s.hc * E, W }));
    RAD_ARCH_TRY(d.f32(key_b, { s.hc * E }));
    RAD_ARCH_TRY(d.f32(key_n, { s.hc * E }));
    RAD_ARCH_TRY(d.f32(query_n, { s.hc * E }));
    RAD_ARCH_TRY(d.mat(value, { E, W }));
    RAD_ARCH_TRY(d.f32(value_b, { E }));
    RAD_ARCH_TRY(d.f32(value_g, { E }));
    RAD_ARCH_TRY(d.f32(branch, { s.hc * E }));
    return RAD_OK;
}

/* A hyper-connection: `fn` scores the flattened streams into 2 * hc logits (the read gate's hc,
 * then the write gate's), `base` their biases, `scale` one factor for each half. f32 in the
 * checkpoint, and the reference applies it in f32 whatever the model's dtype. */
static int hyper(Decl& d, Names& nm, const Shape& s, const char* scope, const char* hp) {
    RadBuilder* b = d.b;
    const char* fn = nm.f("%s.fn", scope);
    const char* base = nm.f("%s.base", scope);
    const char* scale = nm.f("%s.scale", scope);
    RAD_ARCH_TRY(map_copy(b, fn, nm.ckpt("%s.hc_fn", hp)));
    RAD_ARCH_TRY(map_copy(b, base, nm.ckpt("%s.hc_base", hp)));
    RAD_ARCH_TRY(map_copy(b, scale, nm.ckpt("%s.hc_scale", hp)));
    RAD_ARCH_TRY(d.f32(fn, { 2 * s.hc, s.hc * s.n_embd }));
    RAD_ARCH_TRY(d.f32(base, { 2 * s.hc }));
    RAD_ARCH_TRY(d.f32(scale, { 2 }));
    return RAD_OK;
}

static int trunk_layer(RadBuilder* b, Model& m, int l) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    Decl d{ b, m, l, RAD_ACCESS_PER_TOKEN };
    const char* pre = nm.ckpt("model.layers.%d", l);
    const char* ap = nm.ckpt("%s.self_attn", pre);
    const char* mp = nm.ckpt("%s.mlp", pre);
    for (size_t i = 0; i < s.engram_layers.size(); ++i)
        if (s.engram_layers[i] == l) RAD_ARCH_TRY(engram(d, nm, s, l, s.engram_rows[i], pre));

    RAD_ARCH_TRY(hyper(d, nm, s, nm.f("blk.%d.hc_attn", l),
                       nm.ckpt("%s.hc_attn_layer.hc_pre", pre)));
    const char* an = nm.f("blk.%d.attn_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, an, nm.ckpt("%s.input_layernorm.weight", pre)));
    RAD_ARCH_TRY(d.f32(an, { s.n_embd }));
    const bool sliding = s.is_sliding[(size_t)l] != 0;
    RAD_ARCH_TRY(attention(d, nm, s, sliding ? s.sliding : s.sparse, nm.f("blk.%d", l), ap));
    if (!sliding) RAD_ARCH_TRY(indexer(d, nm, s, l, ap));

    RAD_ARCH_TRY(hyper(d, nm, s, nm.f("blk.%d.hc_ffn", l),
                       nm.ckpt("%s.hc_mlp_layer.hc_pre", pre)));
    const char* fnorm = nm.f("blk.%d.ffn_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, fnorm, nm.ckpt("%s.post_attention_layernorm.weight", pre)));
    RAD_ARCH_TRY(d.f32(fnorm, { s.n_embd }));
    if (s.is_moe[(size_t)l]) return moe(d, nm, s, l, mp);
    /* A dense layer's sandwich norm is the layer's (`post_fc2_norm`), a MoE layer's the block's
     * (`combine_output_norm`); one role, one declared name. */
    const char* post = nm.f("blk.%d.ffn_post_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, post, nm.ckpt("%s.post_fc2_norm.weight", mp)));
    RAD_ARCH_TRY(mlp(d, nm, s, nm.f("blk.%d", l), mp, s.n_ff, false));
    RAD_ARCH_TRY(d.f32(post, { s.n_embd }));
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
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS ||
        rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->world_size > 1)
        return refuse("the manifest is declared whole, for one rank; it has no tensor-parallel "
                      "split yet");
    auto& slot = models[rad_build_slot(ctx)];
    slot = std::make_unique<Model>();
    Model& m = *slot;
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    Shape& s = m.s;
    RAD_ARCH_TRY(read_shape(s, meta));

    Decl d{ b, m, -1, RAD_ACCESS_VOCAB };
    const char* token = nm.f("token_embd.weight");
    const char* head = nm.f("output.weight");
    const char* fin = nm.f("output_norm.weight");
    RAD_ARCH_TRY(map_copy(b, token, "model.embed_tokens.weight"));
    /* The release ships its own lm_head (`tie_word_embeddings` false); a tied fine-tune reads
     * the embedding. The reference keeps the head in f32 by name, widened from these bf16 rows. */
    RAD_ARCH_TRY(map_copy_alt(b, head, "lm_head.weight", "model.embed_tokens.weight"));
    RAD_ARCH_TRY(map_copy(b, fin, "model.norm.weight"));
    RAD_ARCH_TRY(d.put(token, RAD_BF16, { s.n_vocab, s.n_embd }, grp_model()));
    RAD_ARCH_TRY(d.put(head, RAD_BF16, { s.n_vocab, s.n_embd }, grp_model()));
    d.access = RAD_ACCESS_PER_TOKEN;
    RAD_ARCH_TRY(d.put(fin, RAD_F32, { s.n_embd }, grp_model()));
    /* The head's collapse of the streams: hc logits, one scale, and no write gate. */
    const char* hfn = nm.f("hc_head.fn");
    const char* hbase = nm.f("hc_head.base");
    const char* hscale = nm.f("hc_head.scale");
    RAD_ARCH_TRY(map_copy(b, hfn, "model.hc_head.hc_head_fn"));
    RAD_ARCH_TRY(map_copy(b, hbase, "model.hc_head.hc_head_base"));
    RAD_ARCH_TRY(map_copy(b, hscale, "model.hc_head.hc_head_scale"));
    RAD_ARCH_TRY(d.put(hfn, RAD_F32, { s.hc, s.hc * s.n_embd }, grp_model()));
    RAD_ARCH_TRY(d.put(hbase, RAD_F32, { s.hc }, grp_model()));
    RAD_ARCH_TRY(d.put(hscale, RAD_F32, { 1 }, grp_model()));

    for (int64_t l = 0; l < s.n_layer; ++l) RAD_ARCH_TRY(trunk_layer(b, m, (int)l));
    /* A container carries what the model can serve, so the drafter and the tower are declared
     * wherever the checkpoint has them and the caller left room (rad-convert's --max-spec 0
     * leaves the drafter out). */
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
             "MLA, %lld latent MoE of %lld experts, %zu Engram tables, DFly2 %s, vision %s; "
             "%lld weights declared",
             (long long)s.n_layer, (long long)s.hc, (long long)n_sparse,
             (long long)(s.n_layer - n_sparse), (long long)n_moe, (long long)s.n_expert,
             s.engram_layers.size(), dfly ? "declared" : "left out", vit ? "declared" : "left out",
             (long long)m.n_weights);
    rad_note(b, "hyv4flash declares the weight manifest and name map only: a container converts "
                "and inspects, and does not serve until the graph is written");
    return RAD_OK;
}

/* Nothing unimplemented returns success: a step is refused by name. */
static void step(RadCtx* c, const RadBatch*) {
    rad_step_fail(c, "hyv4flash declares Hy4-flash's weights for rad-convert and has no serving "
                     "graph yet; this container cannot run a step");
}
} // namespace hyv4flash

#ifndef RAD_ARCH_NO_EXPORTS
/* The manifest's state is indexed by slot (rad_build_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(hyv4flash, "hyv4flash", "", "0.1.0",
                "Tencent Hy4-flash: the weight manifest and checkpoint name map (conversion only)")
RAD_ARCH_CHAT_FORMAT(hyv4flash)
#endif
