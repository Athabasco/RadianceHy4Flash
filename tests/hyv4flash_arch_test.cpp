/* Hy4-flash's declared graph and step (arch/hyv4flash), without a GPU or checkpoint: the harness's
 * `mini` geometry (ref/minicfg.py, the convert test's fixture) at one and two ranks, and its
 * `mini4` twin (twice the heads and experts) at one, two and four ranks, bf16. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/hyv4flash/hyv4flash.cpp"

#include <cstdlib>

using namespace rad_fake;

namespace {

/* The `mini` preset as rad-convert records its config: 3 layers (sparse, sliding, sparse; layer 0
 * dense, 1-2 latent MoE), two streams, an Engram in front of layer 1, dense limit 16. `wide` is
 * `mini4`: 4 sparse and 8 sliding heads and 8 experts, so four ranks hold whole heads. */
MetaKV mini_meta(bool wide = false, int64_t ff_exp = 128) {
    RadModelMeta m{};
    m.arch_id = "hyv4flash";
    m.name = "test-hy4-mini";
    m.quant = "";
    m.n_layers = 3;
    m.n_embd = 128;
    m.n_head = 2;
    m.n_head_kv = 2;
    m.head_dim = 8;
    m.n_ff = 256;
    m.n_vocab = 320;
    m.n_ctx_train = 64;
    m.rms_eps = 1e-5f;
    m.rope_theta = 10000.0f;
    m.n_expert = 4;
    m.n_expert_used = 2;
    MetaKV k(m);
    const char* kv[][2] = {
        { "hc_mult", "2" }, { "q_lora_rank", "64" }, { "kv_lora_rank", "32" },
        { "num_attention_heads", "2" }, { "qk_nope_head_dim", "24" }, { "qk_rope_head_dim", "8" },
        { "v_head_dim", "32" }, { "swa_num_attention_heads", "4" },
        { "swa_qk_nope_head_dim", "24" }, { "swa_qk_rope_head_dim", "8" },
        { "swa_v_head_dim", "32" }, { "sliding_window", "8" },
        { "layer_types", "deepseek_sparse_attention sliding_attention deepseek_sparse_attention" },
        { "mlp_layer_types", "dense sparse sparse" }, { "intermediate_size", "256" },
        { "n_routed_experts", "4" }, { "n_shared_experts", "1" }, { "num_experts_per_tok", "2" },
        { "moe_intermediate_size", "128" }, { "routed_expert_hidden_size", "128" },
        { "routed_scaling_factor", "3.998" }, { "swiglu_limit", "10" }, { "norm_topk_prob", "1" },
        { "index_n_heads", "2" }, { "index_head_dim", "16" },
        { "compressed_indexer_compress_ratio", "4" }, { "compressed_indexer_group_topk", "4" },
        { "use_mla", "1" }, { "gated_mla", "1" }, { "learnable_sink", "1" },
        { "sandwich_norm", "1" }, { "enable_ihc", "1" }, { "use_dsa", "1" },
        { "use_compressed_indexer", "1" }, { "attention_bias", "0" },
        { "moe_router_use_sigmoid", "1" }, { "moe_router_enable_expert_bias", "1" },
        { "hc_eps", "1e-06" }, { "hc_magnitude", "2" },
        { "rope_parameters.sliding_attention.rope_theta", "10000" },
        { "rope_parameters.deepseek_sparse_attention.rope_theta", "10000000" },
        { "engram_layer_ids", "2" }, { "engram_max_ngram_size", "3" },
        { "engram_n_head_per_ngram", "5" }, { "engram_n_embed_per_ngram", "40" },
        { "engram_layer_head_vocab_sizes.2", "11 13 17 19 23 29 31 37 41 43" },
        { "engram_pad_id", "2" }, { "engram_lookup_table_size", "300" },
        { "engram_fp8_embedding", "1" }, { "image_token_id", "120120" },
        { "video_token_id", "120124" },
    };
    for (const auto& p : kv) k.set(p[0], p[1]);
    if (wide) {
        k.m.n_head = k.m.n_head_kv = 4;
        k.m.n_expert = 8;
        k.set("num_attention_heads", "4").set("swa_num_attention_heads", "8");
        k.set("n_routed_experts", "8");
    }
    if (ff_exp != 128) k.set("moe_intermediate_size", std::to_string(ff_exp).c_str());
    return k;
}

/* A bf16 container: the experts plain (MoeFP8 reads the format off the first expert). */
void bf16_encs(RadBuilder* b) {
    b->encs.clear();
    b->encs.push_back({ "ffn_gate_up_exps", rad_enc_plain(RAD_BF16) });
    b->encs.push_back({ "ffn_down_exps", rad_enc_plain(RAD_BF16) });
    b->kv_block_by_consumer = true;
}

/* The quantised forms of PLAN §4.3 (libquant's encodings, as rad-convert records them). */
RadEncoding enc_i8(bool rot) {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    if (rot) rad_enc_copy_str(e.transform, "fwht128");
    return e;
}
/* The codebook at E4M3 scales a 64 of K under a fixed f32: w4nl64, w5nl64, w6nl64. */
RadEncoding enc_nl64(int bits) {
    RadEncoding e = rad_enc_affine(bits == 4 ? RAD_U4 : bits == 5 ? RAD_U5 : RAD_U6, RAD_F8E4M3,
                                   1, 64);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 1 << bits);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}
/* The rotated computed trellis under one f32 scale a row (IQR; tcq_code=mad). */
RadEncoding enc_trellis(uint32_t codes, bool colf) {
    RadEncoding e{};
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "trellis");
    rad_enc_add_plane(&e, "codes", codes, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F32, 1, 0);
    const int t = rad_enc_add_table(&e, "trellis", RAD_I8, 1 << 16, 1);
    e.plane[t].block[0] = 128;
    rad_enc_add_table(&e, "trellis.mad", RAD_U32, 2, 1);
    if (colf) rad_enc_add_plane(&e, "scale.col", RAD_F32, 0, 128);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

struct Declared {
    MetaKV mk;
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    int rc = 0;
    explicit Declared(void (*encs)(RadBuilder*) = bf16_encs, int world = 1, int rank = 0,
                      bool wide = false, int64_t ff_exp = 128)
        : mk(mini_meta(wide, ff_exp)) {
        encs(&b);
        ctx = build_ctx(rank, world);
        std::string err;
        err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        if (rc != RAD_OK && std::getenv("HY4_TEST_STDERR")) fputs(err.c_str(), stderr);
    }
};

const RecKV* kv_named(const RadBuilder& b, const char* name) {
    for (const auto& k : b.kv_groups)
        if (k.name == name) return &k;
    return nullptr;
}

int64_t block_of(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.kv_groups.size(); ++i)
        if (b.kv_groups[i].name == name) return fake_kv_block(&b, (rad_kvgroup)(i + 1));
    return -1;
}

int count_issued(const RadBuilder& b, const RadCtx& c, const char* op) {
    int n = 0;
    for (const auto& s : issued_ops(b, c)) n += s == op;
    return n;
}

}  // namespace

/* WP16.0c / WP16.1a: the whole-on-every-rank paged groups are declared replicated at every world
 * (the host tier keeps each once a slot), the per-rank state groups are not, and the plugin asks
 * for chained launches. */
TEST(the_replicated_kv_groups_and_the_chain_request) {
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            Declared d(bf16_encs, world, rank);
            REQUIRE_EQ(d.rc, RAD_OK);
            for (const char* n : { "kv_latent_sparse", "kv_latent_sliding", "kv_index_keys" }) {
                const RecKV* k = kv_named(d.b, n);
                REQUIRE(k != nullptr);
                CHECK(k->replicated);
            }
            for (const char* n : { "kv_index_tail", "kv_engram_tokens" }) {
                const RecKV* k = kv_named(d.b, n);
                REQUIRE(k != nullptr);
                CHECK(!k->replicated);
            }
        }
    }
    Declared d;
    RadArchProbe p{};
    CHECK_OK(hyv4flash::probe(d.mk.meta(), &p));
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);
}

/* Every layer kind's ops, once each where the plan says so: the connections, both MLA forms on
 * both layer kinds (the sparse ones with the selection form), the indexer on the sparse layers, the
 * dense MLP and the latent MoE, the Engram in front of layer 1, the head. */
TEST(the_mini_graph_declares_every_layer_kind) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const RadBuilder& b = d.b;
    CHECK_EQ(count_op(b, "ihc_pre"), 7);                /* 2 a layer and the head's collapse */
    CHECK_EQ(count_op(b, "ihc_post"), 6);
    CHECK_EQ(count_op(b, "mla_project_kv"), 6);         /* absorb and expand a layer */
    CHECK_EQ(count_op(b, "mla_attn_paged_sink"), 3);
    CHECK_EQ(count_op(b, "mla_attn_sel_sink"), 2);      /* the sparse layers */
    CHECK_EQ(count_op(b, "attn_kv_sink"), 3);
    CHECK_EQ(count_op(b, "mla_kv_split"), 3);
    CHECK_EQ(count_op(b, "gate_mul"), 3);
    CHECK_EQ(count_op(b, "kv_store"), 3);
    CHECK_EQ(count_op(b, "qsa_work"), 2);
    CHECK_EQ(count_op(b, "idx_pool_gated"), 2);
    CHECK_EQ(count_op(b, "qsa_tail_store"), 2);
    CHECK_EQ(count_op(b, "idx_score"), 2);
    CHECK_EQ(count_op(b, "qsa_select"), 2);
    CHECK_EQ(count_op(b, "engram_ids"), 2);             /* the decode and the prefill layout */
    CHECK_EQ(count_op(b, "embed_lookup_q"), 1);
    CHECK_EQ(count_op(b, "quant_act_fp8"), 1);          /* the bf16 table's E4M3 round trip */
    CHECK_EQ(count_op(b, "dequant"), 1);
    CHECK_EQ(count_op(b, "engram_norm"), 1);
    CHECK_EQ(count_op(b, "engram_gate"), 1);
    CHECK_EQ(count_op(b, "router_sigmoid_topk"), 2);
    CHECK_EQ(count_op(b, "swiglu_clamped"), 2);         /* the routed arms; the shared ones are not */
    CHECK_EQ(count_op(b, "logits_gemm"), 1);
    CHECK_EQ(count_op(b, "cast") > 0, true);
    CHECK(find_b(b, "dbg.resid.0") == nullptr);         /* no taps unless asked */

    /* The sink attention's geometry: a sliding layer's window, the scale 1/sqrt(nope + rope). */
    const RecOp* sw = find_op(b, "mla_attn_paged_sink", 1);
    CHECK_EQ(pint(sw, "heads"), 4);
    CHECK_EQ(pint(sw, "window"), 8);
    CHECK_EQ(pint(sw, "block_size"), 16);
    CHECK(std::fabs(pflt(sw, "scale") - 1.0 / std::sqrt(32.0)) < 1e-12);
    const RecOp* sp = find_op(b, "mla_attn_paged_sink", 0);
    CHECK_EQ(pint(sp, "window"), 0);
    CHECK_EQ(pint(sp, "block_size"), 4);
    const RecOp* sel = find_op(b, "qsa_select");
    CHECK_EQ(pint(sel, "topk"), 3);
    CHECK_EQ(pint(sel, "own"), 1);
    CHECK_EQ(pint(sel, "signed"), 1);
    const RecOp* r = find_op(b, "router_sigmoid_topk");
    CHECK_EQ(pint(r, "top_k"), 2);
    CHECK(std::fabs(pflt(r, "route_scale") - 3.998) < 1e-12);
    CHECK(std::fabs(pflt(find_op(b, "swiglu_clamped"), "limit") - 10.0) < 1e-12);
    const RecOp* ih = find_op(b, "ihc_pre");
    CHECK(std::fabs(pflt(ih, "post_mag") - 2.0) < 1e-12);
    CHECK(std::fabs(pflt(ih, "hc_eps") - 1e-6) < 1e-15);
    /* The Engram eps is the bf16 default (DECISIONS U1) unless the switch says f32. */
    CHECK(std::fabs(pflt(find_op(b, "engram_norm"), "eps") - 0.0078125) < 1e-12);

    /* The caches (PLAN §1.3). */
    const RecKV* swk = kv_named(b, "kv_latent_sliding");
    const RecKV* spk = kv_named(b, "kv_latent_sparse");
    const RecKV* ixk = kv_named(b, "kv_index_keys");
    const RecKV* tl = kv_named(b, "kv_index_tail");
    const RecKV* eg = kv_named(b, "kv_engram_tokens");
    REQUIRE(swk && spk && ixk && tl && eg);
    CHECK_EQ(swk->d.kind, RAD_KV_WINDOW);
    CHECK_EQ(swk->d.window, 8);
    CHECK_EQ(swk->d.head_dim, 20);                      /* (32 + 8) / 2 */
    CHECK_EQ(block_of(b, "kv_latent_sliding"), 16);
    CHECK_EQ(spk->d.kind, RAD_KV_FULL);
    CHECK_EQ(block_of(b, "kv_latent_sparse"), 4);
    CHECK_EQ(block_of(b, "kv_index_keys"), 4);
    CHECK_EQ(ixk->d.head_dim, 2);                       /* a page is one 16-wide key */
    CHECK_EQ(tl->d.kind, RAD_KV_LINEAR);
    CHECK_EQ(tl->d.state_dim[0], 4);
    CHECK_EQ(eg->d.kind, RAD_KV_CONV);
    CHECK_EQ(eg->d.dtype, RAD_I32);

    /* The weights the graph reads under the names it declares them (PLAN D10: the router is
     * GlmRouter's). kv_b is plain bf16, the router f32. */
    CHECK(find_w(b, "blk.1.glm_router.weight") != nullptr);
    CHECK(find_w(b, "blk.1.glm_router.correction") != nullptr);
    CHECK(find_w(b, "blk.1.ffn_gate_inp.weight") == nullptr);
    CHECK_EQ(find_w(b, "blk.1.glm_router.weight")->d.dtype, RAD_F32);
    CHECK_EQ(find_w(b, "blk.0.attn_kv_b.weight")->d.dtype, RAD_BF16);
    CHECK(find_w(b, "blk.0.attn_q_nope.weight") != nullptr);
    CHECK(find_w(b, "blk.0.attn_q_rope.weight") != nullptr);
    CHECK(find_w(b, "blk.0.index_query_nope.weight") != nullptr);
    CHECK(find_w(b, "blk.1.index_query_nope.weight") == nullptr);
    CHECK(find_w(b, "blk.1.engram_mult") != nullptr);
    /* q_b's rows per head: nope then rope, a piece a head. */
    const RecMap* qn = find_map(b, "blk.1.attn_q_nope.weight");
    REQUIRE(qn != nullptr);
    CHECK_EQ(qn->src.size(), (size_t)4);
    CHECK_EQ(qn->src_index[1], 32);
}

/* The steps the core builds, each lint clean (liveness by declaration order, spans, KV binds):
 * decode past the dense limit (the selection), a fresh prefill below it (non-absorbed on every
 * layer), a mixed step, a chunk with a cached prefix (absorbed), and the absorbed switch. */
TEST(every_step_kind_is_lint_clean) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    RadBuilder& b = d.b;
    struct Case { const char* what; std::vector<int> dq, pq; int32_t ctx; bool host_pos; };
    const Case cases[] = {
        { "decode", { 1, 1, 1 }, {}, 0, true },
        { "fresh prefill", {}, { 12 }, 0, true },
        { "fresh prefill, long", {}, { 40 }, 0, true },
        { "mixed", { 1 }, { 10 }, 0, true },
        { "prefix chunk", {}, { 16 }, 16, true },
        { "no host positions", {}, { 12 }, 0, false },
    };
    for (const Case& cs : cases) {
        auto k = kit(b, cs.dq, cs.pq, cs.ctx);
        if (cs.host_pos) k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(b, &c);
        if (!c.step_fail.empty()) fprintf(stderr, "%s: %s\n", cs.what, c.step_fail.c_str());
        CHECK(c.step_fail.empty());
        const int dense = count_issued(b, c, "attn_kv_sink");
        const int sel = count_issued(b, c, "mla_attn_sel_sink");
        if (!std::strcmp(cs.what, "fresh prefill")) {
            CHECK_EQ(dense, 3);       /* every layer non-absorbed */
            CHECK_EQ(sel, 0);
            CHECK_EQ(count_issued(b, c, "idx_score"), 0);
        } else if (!std::strcmp(cs.what, "fresh prefill, long")) {
            CHECK_EQ(dense, 1);       /* the sliding layer; the sparse ones select */
            CHECK_EQ(sel, 2);
        } else if (!std::strcmp(cs.what, "decode")) {
            CHECK_EQ(dense, 0);
            CHECK_EQ(sel, 2);         /* contexts of 40 and more pass the limit of 16 */
        } else if (!std::strcmp(cs.what, "prefix chunk") ||
                   !std::strcmp(cs.what, "no host positions")) {
            CHECK_EQ(dense, 0);       /* a cached prefix (or none known) is absorbed */
        }
    }
    setenv("RADIANCE_HY4_PREFILL", "absorbed", 1);
    Declared a;
    unsetenv("RADIANCE_HY4_PREFILL");
    REQUIRE_EQ(a.rc, RAD_OK);
    auto k = kit(a.b, {}, { 12 }, 0);
    k->b.positions_h = k->pos.data();
    RadCtx c{ .b = &a.b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    CHECK_LINT(a.b, &c);
    CHECK_EQ(count_issued(a.b, c, "attn_kv_sink"), 0);
}

/* RADIANCE_HY4_TAPS=1: the harness's buffers (ref/README.md "Tap contract"), PERSIST, and a step
 * that writes them stays lint clean. */
TEST(the_debug_taps_are_persistent_buffers_and_lint_clean) {
    setenv("RADIANCE_HY4_TAPS", "1", 1);
    Declared d;
    unsetenv("RADIANCE_HY4_TAPS");
    REQUIRE_EQ(d.rc, RAD_OK);
    for (const char* n : { "dbg.resid.0", "dbg.resid_mid.2", "dbg.attn_hc_x.1", "dbg.attn_hc_post.0",
                           "dbg.attn_out.1", "dbg.mlp_out.0", "dbg.engram_out.1",
                           "dbg.router_topk_idx.1", "dbg.index_token_keys.0",
                           "dbg.index_group_keys.2", "dbg.index_sel.0", "dbg.index_nsel.2", "dbg.index_score.0",
                           "dbg.embed", "dbg.final_hidden" }) {
        const RecBuf* t = find_b(d.b, n);
        CHECK(t != nullptr);
        if (t) CHECK_EQ(t->d.kind, RAD_BUF_PERSIST);
    }
    CHECK(find_b(d.b, "dbg.engram_out.0") == nullptr);
    CHECK(find_b(d.b, "dbg.index_sel.1") == nullptr);
    CHECK_EQ(find_b(d.b, "dbg.router_topk_idx.1")->d.dtype, RAD_I32);
    for (const auto& pq : { std::vector<int>{ 12 }, std::vector<int>{ 40 } }) {
        auto k = kit(d.b, {}, pq, 0);
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(d.b, &c);
        CHECK(c.step_fail.empty());
    }
}

/* What this build does not serve is refused by name at declare: a quantised kv_b (DECISIONS U2),
 * a recipe over a parameter the model applies in f32, a quantised token embedding, and a trunk
 * linear in a form that reads E4M3 codes no producer of this graph writes (MXFP4). */
TEST(unserved_forms_are_refused_at_declare) {
    auto kvb = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "attn_kv_b", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
    };
    CHECK_EQ(Declared(kvb).rc, RAD_E_UNSUPPORTED);
    auto experts = [](RadBuilder* b) {
        bf16_encs(b);
        RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        rad_enc_copy_str(i8.transform, "fwht128");
        b->encs[0].second = i8;
        b->encs[1].second = i8;
    };
    auto hc = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "hc_attn.fn", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
    };
    CHECK(Declared(hc).rc < 0);
    auto tok = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "token_embd", enc_i8(false) });
    };
    CHECK_EQ(Declared(tok).rc, RAD_E_UNSUPPORTED);
    auto mx = [](RadBuilder* b) {
        bf16_encs(b);
        RadEncoding e = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
        b->encs.insert(b->encs.begin(), { "attn_q_a", e });
    };
    CHECK_EQ(Declared(mx).rc, RAD_E_DTYPE);
}


/* ------------------------------------------------------------------ tensor parallelism (PLAN §1.5) */
namespace {

/* What a weight's name says it is: an attention part split by whole heads, a width split, the
 * vocabulary-split head, a routed expert (whole on one rank), or replicated. */
enum Part { kReplicated, kRows, kCols, kExpert };
Part part_of(const std::string& n) {
    auto has = [&](const char* x) { return n.find(x) != std::string::npos; };
    if (has("_exps.")) return kExpert;
    if (n.rfind("dfly.", 0) == 0 || n.rfind("v.", 0) == 0) return kReplicated;   /* not served */
    for (const char* r : { ".attn_q_nope.", ".attn_q_rope.", ".attn_kv_b.", ".attn_gate.",
                           ".attn_sinks.", ".ffn_gate_up.", ".ffn_gate_up_shexp." })
        if (has(r)) return kRows;
    if (n == "output.weight") return kRows;
    for (const char* c : { ".attn_output.", ".ffn_down.", ".ffn_down_shexp." })
        if (has(c)) return kCols;
    return kReplicated;
}

rad_buf buf_named(const RadBuilder& b, const char* n) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == n) return (rad_buf)(i + 1);
    return 0;
}

void steps_clean_at(RadBuilder& b, int rank, int world) {
    struct Case { std::vector<int> dq, pq; int32_t ctx; };
    const Case cases[] = { { { 1, 1, 1 }, {}, 0 }, { {}, { 12 }, 0 }, { {}, { 40 }, 0 },
                           { { 1 }, { 10 }, 0 }, { {}, { 16 }, 16 } };
    for (const Case& cs : cases) {
        auto k = kit(b, cs.dq, cs.pq, cs.ctx);
        k->b.positions_h = k->pos.data();
        auto v = run_passes(b, hyv4flash::step, { k.get() }, rank, world);
        for (auto& c : v)
            if (!c->step_fail.empty()) fprintf(stderr, "rank %d/%d: %s\n", rank, world,
                                               c->step_fail.c_str());
        CHECK_PASSES(b, v);
    }
}

}  // namespace

/* Every weight of the one-rank graph is covered exactly once by the ranks of two and four: a head
 * or width split is an even share along its split dimension on every rank (each stacked half of
 * a fused gate_up on its own), a routed expert is whole on exactly one rank (bf16 experts,
 * `expert_base = rank * E / world`), and everything else (iHC, norms, q_a, kv_a, the indexer,
 * the router, the latent pair, the Engram, the embedding) is the whole weight on every rank. */
TEST(the_ranks_cover_every_weight_exactly_once) {
    for (bool wide : { false, true }) {
        Declared one(bf16_encs, 1, 0, wide);
        REQUIRE_EQ(one.rc, RAD_OK);
        std::map<std::string, const RecWeight*> whole;
        for (const auto& w : one.b.weights) whole[w.name] = &w;
        for (int world : { 2, 4 }) {
            if (!wide && world == 4) continue;   /* mini's 2 sparse heads: refused, below */
            std::vector<std::unique_ptr<Declared>> ranks;
            std::map<std::string, int> seen;
            for (int r = 0; r < world; ++r) {
                ranks.push_back(std::make_unique<Declared>(bf16_encs, world, r, wide));
                REQUIRE_EQ(ranks.back()->rc, RAD_OK);
                for (const auto& w : ranks.back()->b.weights) {
                    ++seen[w.name];
                    auto it = whole.find(w.name);
                    REQUIRE(it != whole.end());
                    const RadWeightDecl& o = it->second->d;
                    const Part pt = part_of(w.name);
                    CHECK_EQ(w.d.rank, o.rank);
                    CHECK(!w.span_set);   /* every split here is even */
                    const int split = pt == kRows ? 0 : pt == kCols ? 1 : -1;
                    CHECK_EQ(w.d.shard, pt == kRows ? (int)RAD_SHARD_ROW
                                        : pt == kCols ? (int)RAD_SHARD_COL
                                                      : (int)RAD_SHARD_NONE);
                    for (uint32_t k = 0; k < o.rank; ++k) {
                        const int64_t want = (int)k == split ? o.shape[k] / world : o.shape[k];
                        if (w.d.shape[k] != want)
                            fprintf(stderr, "%s dim %u: %lld, want %lld (world %d)\n",
                                    w.name.c_str(), k, (long long)w.d.shape[k], (long long)want,
                                    world);
                        CHECK_EQ(w.d.shape[k], want);
                    }
                    /* A fused gate_up splits each half (an expert is whole: no parts). */
                    if (pt == kExpert) continue;
                    CHECK_EQ(w.d.n_row_parts, o.n_row_parts);
                    for (int q = 0; q < o.n_row_parts; ++q)
                        CHECK_EQ(w.d.row_parts[q], o.row_parts[q] / (pt == kRows ? world : 1));
                }
            }
            for (const auto& [name, w] : whole) {
                const int want = part_of(name) == kExpert ? 1 : world;
                if (seen[name] != want)
                    fprintf(stderr, "%s on %d of %d ranks, want %d\n", name.c_str(), seen[name],
                            world, want);
                CHECK_EQ(seen[name], want);
            }
            /* Experts [r E / W, (r + 1) E / W) on rank r. */
            const int E = wide ? 8 : 4;
            for (int r = 0; r < world; ++r)
                for (int e = 0; e < E; ++e) {
                    char n[64];
                    std::snprintf(n, sizeof n, "blk.2.ffn_down_exps.%d.weight", e);
                    CHECK_EQ(find_w(ranks[(size_t)r]->b, n) != nullptr, e / (E / world) == r);
                }
        }
    }
}

/* A rank's graph: its heads in every attention op, two exact all-reduces a layer (behind o_proj and
 * behind the feed-forward's sum, each before its sandwich norm, in place), its experts in the
 * routed arm, its vocabulary in the head and the logits buffer; every step kind lint clean on every
 * rank. */
TEST(each_rank_declares_its_share_and_reduces_twice_a_layer) {
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            Declared d(bf16_encs, world, rank, true);
            REQUIRE_EQ(d.rc, RAD_OK);
            RadBuilder& b = d.b;
            const int Hs = 4 / world, Hw = 8 / world, E = 8 / world;
            for (const char* op : { "mla_attn_paged_sink", "mla_project_kv", "attn_kv_sink",
                                    "mla_kv_split" })
                for (int i = 0; i < count_op(b, op); ++i) {
                    const int64_t h = pint(find_op(b, op, i), "heads");
                    CHECK(h == Hs || h == Hw);
                }
            CHECK_EQ(pint(find_op(b, "mla_attn_paged_sink", 0), "heads"), Hs);   /* layer 0 sparse */
            CHECK_EQ(pint(find_op(b, "mla_attn_paged_sink", 1), "heads"), Hw);   /* layer 1 sliding */
            CHECK_EQ(pint(find_op(b, "mla_attn_sel_sink", 0), "heads"), Hs);
            CHECK_EQ(pint(find_op(b, "gate_mul", 1), "n"), Hw * 32);
            /* Replicated: the cache row, the indexer, the router over every expert. */
            CHECK_EQ(kv_named(b, "kv_latent_sparse")->d.head_dim, 20);
            CHECK_EQ(pint(find_op(b, "idx_score"), "heads"), 2);
            CHECK_EQ(pint(find_op(b, "router_sigmoid_topk"), "n_expert"), 8);
            int based = 0;
            for (const auto& o : b.ops)
                if (param(&o, "expert_base")) {
                    ++based;
                    CHECK_EQ(pint(&o, "expert_base"), rank * E);
                }
            CHECK(based >= 2);
            int mg = 0;
            for (const auto& o : b.ops)
                if (o.op == "moe_gemm") {
                    ++mg;
                    CHECK_EQ(pint(&o, "n_expert"), E);
                }
            CHECK_EQ(mg, 4);
            CHECK_EQ(pint(find_op(b, "logits_gemm"), "n_vocab"), 320 / world);
            CHECK_EQ(find_b(b, "logits")->d.shape[1], 320 / world);

            const int nar = count_op(b, "all_reduce");
            CHECK_EQ(nar, world == 1 ? 0 : 6);
            for (int i = 0; i < nar; ++i) {
                const RecOp* ar = find_op(b, "all_reduce", i);
                CHECK_EQ(pint(ar, "exact"), 1);
                CHECK_EQ(pint(ar, "world_size"), world);
                CHECK_EQ(pstr(ar, "dtype"), std::string("bf16"));
                REQUIRE_EQ(ar->reads.size(), (size_t)1);
                CHECK(ar->writes == ar->reads);
                /* Its buffer is what the next rmsnorm reads: o_proj's output or the block's sum. */
                const rad_buf x = ar->reads[0];
                bool normed = false;
                for (const auto& o : b.ops)
                    if (o.op == "rmsnorm" && in(o.reads, x)) normed = true;
                CHECK(normed);
            }
            if (world > 1) {
                const rad_buf o = buf_named(b, "sparse.o_proj"), sum = buf_named(b, "moe.routed"),
                              dense = buf_named(b, "dense_out");
                int on_o = 0, on_sum = 0, on_dense = 0;
                for (int i = 0; i < nar; ++i) {
                    const rad_buf x = find_op(b, "all_reduce", i)->reads[0];
                    on_o += x == o;
                    on_sum += x == sum;
                    on_dense += x == dense;
                }
                CHECK_EQ(on_o, 2);       /* the two sparse layers */
                CHECK_EQ(on_sum, 2);     /* the two MoE layers */
                CHECK_EQ(on_dense, 1);   /* layer 0 */
            }
            steps_clean_at(b, rank, world);
        }
}

/* The taps the harness compares are the replicated values at every world; a rank's partial sums
 * (the gate logits of its heads, the arms' outputs) are tapped at one rank only, o_proj after its
 * reduction. */
TEST(taps_at_more_ranks_are_the_replicated_values) {
    setenv("RADIANCE_HY4_TAPS", "1", 1);
    Declared d(bf16_encs, 2, 1);
    unsetenv("RADIANCE_HY4_TAPS");
    REQUIRE_EQ(d.rc, RAD_OK);
    for (const char* n : { "dbg.resid.0", "dbg.attn_out.1", "dbg.mlp_out.2", "dbg.attn_o_proj.0",
                           "dbg.router_topk_idx.1", "dbg.index_sel.0", "dbg.engram_out.1" })
        CHECK(find_b(d.b, n) != nullptr);
    for (const char* n : { "dbg.attn_gate_logits.0", "dbg.moe_experts_out.1",
                           "dbg.moe_routed_out.1", "dbg.moe_shared_out.2" })
        CHECK(find_b(d.b, n) == nullptr);
    steps_clean_at(d.b, 1, 2);
}

/* What does not split is refused by name at declare: a world other than 1, 2 or 4, heads that are
 * not whole a rank (mini's two sparse heads over four), quantised experts whose width is not whole
 * 128-column groups a rank (mini's 128 at two), and a library without an exact all-reduce. */
TEST(tensor_parallel_refusals) {
    CHECK_EQ(Declared(bf16_encs, 3, 0, true).rc, RAD_E_UNSUPPORTED);
    CHECK_EQ(Declared(bf16_encs, 8, 0, true).rc, RAD_E_UNSUPPORTED);
    CHECK_EQ(Declared(bf16_encs, 4, 0, false).rc, RAD_E_UNSUPPORTED);
    auto experts = [](RadBuilder* b) {
        bf16_encs(b);
        RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        rad_enc_copy_str(i8.transform, "fwht128");
        b->encs[0].second = i8;
        b->encs[1].second = i8;
    };
    CHECK_EQ(Declared(experts, 2).rc, RAD_E_UNSUPPORTED);
    auto no_ar = [](RadBuilder* b) {
        bf16_encs(b);
        b->refuse.insert("all_reduce");
    };
    CHECK_EQ(Declared(no_ar, 2).rc, RAD_E_UNSUPPORTED);
    CHECK_EQ(Declared(no_ar, 1).rc, RAD_OK);   /* one rank reduces nothing */
}

/* ------------------------------------------------------------------ quantised formats (PLAN §4.3) */
namespace {

const RecOp* op_writing(const RadBuilder& b, rad_buf x, const char* name) {
    for (const auto& o : b.ops)
        if (o.op == name && in(o.writes, x)) return &o;
    return nullptr;
}

/* Every declared weight of the served graph (all but the drafter and the tower, declared and not
 * served) is a weight operand of some op: no plane is declared and left unread. */
void every_weight_read(const RadBuilder& b) {
    std::vector<bool> read(b.weights.size() + 1, false);
    for (const auto& o : b.ops)
        for (rad_weight w : o.w)
            if (w > 0 && (size_t)w <= b.weights.size()) read[(size_t)w] = true;
    for (size_t i = 0; i < b.weights.size(); ++i) {
        const std::string& n = b.weights[i].name;
        if (n.rfind("dfly.", 0) == 0 || n.rfind("v.", 0) == 0) continue;
        if (!read[i + 1]) fprintf(stderr, "declared and not read: %s\n", n.c_str());
        CHECK(read[i + 1]);
    }
}

/* Layer 1's routed arm in the form `gu` / `dn`: the GEMMs' dtypes, and the clamped product's
 * chain (swiglu_clamped, then the quantiser the down projection reads). */
void check_routed(const RadBuilder& b, const char* gu_dt, const char* dn_dt, const char* quant) {
    int n = 0;
    for (const auto& o : b.ops)
        if (o.op == "moe_gemm_q") {
            CHECK_EQ(pstr(&o, "dtype"), std::string(n % 2 == 0 ? gu_dt : dn_dt));
            ++n;
        }
    CHECK_EQ(n, 4);   /* two MoE layers, gate_up and down */
    CHECK_EQ(count_op(b, "moe_gemm"), 0);
    CHECK_EQ(count_op(b, "swiglu_clamped"), 2);
    const RecOp* cl = find_op(b, "swiglu_clamped");
    REQUIRE(cl != nullptr);
    CHECK_EQ(pflt(cl, "limit"), 10.0);
    REQUIRE_EQ(cl->writes.size(), (size_t)1);
    const rad_buf prod = cl->writes[0];
    bool quantised = false;
    for (const auto& o : b.ops)
        if (o.op == quant && in(o.reads, prod)) quantised = true;
    CHECK(quantised);
}

}  // namespace

/* Every routed-expert form of PLAN §4.3 at K 1536's shape class (here 128), clamped at 10 in every
 * layer: the codebook forms on the rotated E4M3 arm (clamp, then had_quant_act_fp8 into the down
 * projection's codes; the latent's rotated copy by had_quant_act_fp8), int8 and the IQR trellis on
 * the rotated int8 arm (had_quant_act_i8g), and a trellis gate_up with a six-bit codebook down,
 * whose clamped product is the rotated E4M3 pair beside the int8 gate_up input. Every step kind is
 * lint clean. */
TEST(quantised_routed_experts_are_served_clamped) {
    struct Form {
        const char* what;
        void (*encs)(RadBuilder*);
        const char* gu, *dn, *quant, *hq;
    };
    const Form forms[] = {
        { "w4nl64", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_nl64(4); },
          "w4nl64a8h", "w4nl64a8h", "had_quant_act_fp8", "had_quant_act_fp8" },
        { "w5nl64", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_nl64(5); },
          "w5nl64a8h", "w5nl64a8h", "had_quant_act_fp8", "had_quant_act_fp8" },
        { "w6nl64", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_nl64(6); },
          "w6nl64a8h", "w6nl64a8h", "had_quant_act_fp8", "had_quant_act_fp8" },
        { "i8", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_i8(true); },
          "i8a8h", "i8a8h", "had_quant_act_i8g", "had_quant_act_i8g" },
        { "iqr", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_trellis(RAD_U4, true); },
          "i8a8h", "i8a8h", "had_quant_act_i8g", "had_quant_act_i8g" },
        { "iqr+w6", [](RadBuilder* b) {
              bf16_encs(b);
              b->encs[0].second = enc_trellis(RAD_U4, true);
              b->encs[1].second = enc_nl64(6);
          }, "i8a8h", "w6nl64a8h", "had_quant_act_fp8", "had_quant_act_i8g" },
    };
    for (const Form& f : forms) {
        fprintf(stderr, "form %s\n", f.what);
        Declared d(f.encs);
        REQUIRE_EQ(d.rc, RAD_OK);
        check_routed(d.b, f.gu, f.dn, f.quant);
        /* The latent's rotated codes for gate_up, from latent_in's bf16 output. */
        const rad_buf lat = buf_named(d.b, "moe.latent");
        bool hq = false;
        for (const auto& o : d.b.ops)
            if (o.op == f.hq && in(o.reads, lat)) hq = true;
        CHECK(hq);
        every_weight_read(d.b);
        steps_clean_at(d.b, 0, 1);
    }
}

/* Trunk linears a recipe made int8 (rotated or not) or the u5 trellis read int8 codes each writes
 * itself from its input's bf16 plane into the rank's one scratch pair; the others stay bf16 GEMMs,
 * and an int8 lm_head is read by logits_gemm at its codes' dtype. */
TEST(quantised_trunk_linears_quantise_their_own_input) {
    auto encs = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { { "attn_output", rad_enc_plain(RAD_BF16) },
                                          { "attn_q_a.", enc_trellis(RAD_U5, true) },
                                          { "attn_gate.", enc_i8(true) },
                                          { "ffn_gate_up_shexp", enc_i8(true) },
                                          { "ffn_down_shexp", enc_i8(true) },
                                          { "ffn_gate_up.", enc_i8(true) },
                                          { "ffn_down.", enc_i8(false) },
                                          { "ffn_latent_in", enc_i8(true) },
                                          { "ffn_latent_out", enc_i8(true) },
                                          { "output.weight", enc_i8(false) } });
    };
    Declared d(encs);
    REQUIRE_EQ(d.rc, RAD_OK);
    const RadBuilder& b = d.b;
    /* A pair a width: the 128-wide inputs and the dense down's 256. */
    for (const char* n : { "trunk_codes.128.128.q8", "trunk_codes.256.128.q8" }) {
        const RecBuf* q8 = find_b(b, n);
        REQUIRE(q8 != nullptr);
        CHECK_EQ(q8->d.dtype, RAD_I8);
    }
    CHECK(find_b(b, "trunk_codes.64.128.q8") == nullptr);
    int i8h = 0, i8 = 0;
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt_q") (pstr(&o, "dtype") == "i8a8h" ? i8h : i8) += 1;
    /* q_a and gate (3 layers each), the dense gate_up, the shared pair and the latent pair (2
     * MoE layers each): rotated; the dense down: plain. */
    CHECK_EQ(i8h, 3 + 3 + 1 + 2 * 2 + 2 * 2);
    CHECK_EQ(i8, 1);
    CHECK_EQ(count_op(b, "had_quant_act_i8g") >= i8h, true);
    CHECK_EQ(count_op(b, "quant_act_i8g"), 1);
    /* Left bf16: q_nope / q_rope, kv_a, attn_output, the indexer (their K or rows do not take
     * int8 at mini's widths, or no recipe touched them). */
    CHECK(count_op(b, "gemm_nt") > 0);
    const RecWeight* head = find_w(b, "output.weight");
    REQUIRE(head != nullptr);
    CHECK_EQ(head->d.dtype, RAD_I8);
    every_weight_read(d.b);
    steps_clean_at(d.b, 0, 1);
}

/* The Engram table as E4M3 rows under a bf16 or an f32 scale a row (hy4-flash-bf16.recipe,
 * hy4-flash-bf16-f32.recipe): the scale view at the stored dtype, no runtime round trip. */
TEST(the_engram_table_takes_a_bf16_or_an_f32_row_scale) {
    auto bf = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { { "engram_table", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 1, 0) },
                                          { "engram_mult", rad_enc_plain(RAD_I64) } });
    };
    auto f32 = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { { "engram_table", rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 0) },
                                          { "engram_mult", rad_enc_plain(RAD_I64) } });
    };
    for (auto encs : { +bf, +f32 }) {
        Declared d(encs);
        REQUIRE_EQ(d.rc, RAD_OK);
        const RecWeight* sc = find_w(d.b, "blk.1.engram_table.scale");
        REQUIRE(sc != nullptr);
        CHECK_EQ(sc->d.dtype, encs == +f32 ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16);
        CHECK_EQ(count_op(d.b, "dequant"), 0);
        every_weight_read(d.b);
        steps_clean_at(d.b, 0, 1);
    }
}

/* Quantised experts at two and four ranks are a slice of every expert's width (PLAN §1.5, 768 /
 * 384 at the real size; 256 / 128 here at a 512-wide expert): every rank declares all eight
 * experts, gate_up split by rows in its two parts and down by columns, the routed arm reduced
 * with the shared expert's; every step kind lint clean on every rank. */
TEST(quantised_experts_are_sliced_at_more_ranks) {
    auto nl = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = b->encs[1].second = enc_nl64(4);
    };
    auto i8 = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = b->encs[1].second = enc_i8(true);
    };
    for (auto encs : { +nl, +i8 })
        for (int world : { 2, 4 })
            for (int rank = 0; rank < world; ++rank) {
                Declared d(encs, world, rank, true, 512);
                REQUIRE_EQ(d.rc, RAD_OK);
                const RadBuilder& b = d.b;
                const int64_t F = 512 / world;
                for (int e = 0; e < 8; ++e) {
                    char n[64];
                    std::snprintf(n, sizeof n, "blk.2.ffn_gate_up_exps.%d.weight", e);
                    const RecWeight* gu = find_w(b, n);
                    REQUIRE(gu != nullptr);
                    CHECK_EQ(gu->d.shape[0], 2 * F);
                    CHECK_EQ(gu->d.shard, (int)RAD_SHARD_ROW);
                    CHECK_EQ(gu->d.n_row_parts, 2);
                    std::snprintf(n, sizeof n, "blk.2.ffn_down_exps.%d.weight", e);
                    const RecWeight* dn = find_w(b, n);
                    REQUIRE(dn != nullptr);
                    CHECK_EQ(dn->d.shape[1], F);
                    CHECK_EQ(dn->d.shard, (int)RAD_SHARD_COL);
                }
                for (const auto& o : b.ops)
                    if (o.op == "moe_gemm_q") CHECK_EQ(pint(&o, "n_expert"), 8);
                CHECK_EQ(count_op(b, "all_reduce"), 6);
                every_weight_read(d.b);
                steps_clean_at(d.b, rank, world);
            }
    /* mini's 128-wide expert does not split into whole 128-column groups. */
    CHECK_EQ(Declared(nl, 2, 0, true).rc, RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()
