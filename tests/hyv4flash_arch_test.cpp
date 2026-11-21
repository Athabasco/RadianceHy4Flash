/* Hy4-flash's declared graph and step (arch/hyv4flash), without a GPU or checkpoint: the harness's
 * `mini` geometry (ref/minicfg.py, the convert test's fixture), one rank, bf16. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/hyv4flash/hyv4flash.cpp"

#include <cstdlib>

using namespace rad_fake;

namespace {

/* The `mini` preset as rad-convert records its config: 3 layers (sparse, sliding, sparse; layer 0
 * dense, 1-2 latent MoE), two streams, an Engram in front of layer 1, dense limit 16. */
MetaKV mini_meta() {
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
    return k;
}

/* A bf16 container: the experts plain (MoeFP8 reads the format off the first expert). */
void bf16_encs(RadBuilder* b) {
    b->encs.clear();
    b->encs.push_back({ "ffn_gate_up_exps", rad_enc_plain(RAD_BF16) });
    b->encs.push_back({ "ffn_down_exps", rad_enc_plain(RAD_BF16) });
    b->kv_block_by_consumer = true;
}

struct Declared {
    MetaKV mk = mini_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    int rc = 0;
    explicit Declared(void (*encs)(RadBuilder*) = bf16_encs, int world = 1) {
        encs(&b);
        ctx = build_ctx(0, world);
        rc = stderr_of_rc([&] { return hyv4flash::declare(&b, mk.meta(), &ctx); });
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
                           "dbg.index_group_keys.2", "dbg.index_sel.0", "dbg.index_nsel.2",
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

/* What this build does not serve is refused by name: at declare more ranks, a quantised kv_b and a
 * recipe over a parameter the model applies in f32; at the step a quantised expert. */
TEST(unserved_forms_are_refused_at_declare) {
    CHECK_EQ(Declared(bf16_encs, 2).rc, RAD_E_UNSUPPORTED);
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
    {
        /* Quantised experts declare (rad-convert reads the manifest) and are refused at the step. */
        Declared q(experts);
        CHECK_EQ(q.rc, RAD_OK);
        CHECK(find_w(q.b, "blk.2.ffn_gate_up_exps.3.weight") != nullptr);
        CHECK(find_w(q.b, "blk.2.glm_router.weight") != nullptr);
        auto k = kit(q.b, { 1 }, {}, 0);
        RadCtx c{ .b = &q.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.find("quantised") != std::string::npos);
    }
    auto hc = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "hc_attn.fn", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
    };
    CHECK(Declared(hc).rc < 0);
}

RAD_TEST_MAIN()
