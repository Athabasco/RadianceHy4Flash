/* A small Hy4-flash checkpoint, with the release's tensor names, converted through
 * arch/hyv4flash's name map: plain, and by the two recipes data/recipes/ has for the model. */
#include "rad_test.h"
#include "tiny_model.h"
#include "format/radfile.h"
#include "text/tokenizer.h"

#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

using namespace rad;

namespace {
/* One geometry of each kind the plugin reads from the config, none of them the release's: two
 * streams, two sparse layers around a sliding one, layer 0 dense and two latent MoE layers, an
 * Engram of ten hash heads in front of layer 1 (so its per-head norms take two maps), one drafter
 * layer whose MLP width only its tensor states, and a one-block tower. */
constexpr int H = 128, V = 320, HC = 2, QL = 64, KVL = 32, NOPE = 24, ROPE = 8, VD = 32;
constexpr int HEADS = 2, SWA_HEADS = 4, FF = 256, NE = 4, FE = 128, LAT = 128;
constexpr int IXH = 2, IXD = 16, POOL = 4, MAP = 300;
constexpr int EHEADS = 10, EHD = 8, EROWS = 264; /* 11+13+17+19+23 + 29+31+37+41+43 */
constexpr int DFF = 192, VE = 32, VF = 64, VP = 4, VM = 2, VPOS = 17;

struct Fixture {
    tiny::TempDir dir{ "hyv4flash_convert" };
    std::string ckpt;
    std::map<std::string, std::string> src;
    bool ok = false;

    Fixture() {
        if (!dir.ok()) return;
        ckpt = dir / "model";
        if (!std::filesystem::create_directory(ckpt)) return;
        Json text = {
            { "model_type", "hy_v4_flash" }, { "hidden_size", H }, { "num_hidden_layers", 3 },
            { "vocab_size", V }, { "max_position_embeddings", 64 }, { "rms_norm_eps", 1e-5 },
            { "hc_mult", HC }, { "q_lora_rank", QL }, { "kv_lora_rank", KVL },
            { "num_attention_heads", HEADS }, { "qk_nope_head_dim", NOPE },
            { "qk_rope_head_dim", ROPE }, { "v_head_dim", VD },
            { "swa_num_attention_heads", SWA_HEADS }, { "swa_qk_nope_head_dim", NOPE },
            { "swa_qk_rope_head_dim", ROPE }, { "swa_v_head_dim", VD }, { "sliding_window", 8 },
            { "layer_types", { "deepseek_sparse_attention", "sliding_attention",
                               "deepseek_sparse_attention" } },
            { "mlp_layer_types", { "dense", "sparse", "sparse" } },
            { "intermediate_size", FF }, { "n_routed_experts", NE }, { "n_shared_experts", 1 },
            { "num_experts_per_tok", 2 }, { "moe_intermediate_size", FE },
            { "routed_expert_hidden_size", LAT }, { "index_n_heads", IXH },
            { "index_head_dim", IXD }, { "compressed_indexer_compress_ratio", POOL },
            { "use_mla", true }, { "gated_mla", true }, { "learnable_sink", true },
            { "sandwich_norm", true }, { "enable_ihc", true }, { "use_dsa", true },
            { "use_compressed_indexer", true }, { "attention_bias", false },
            { "moe_router_use_sigmoid", true }, { "moe_router_enable_expert_bias", true },
            { "engram_layer_ids", { 2 } }, { "engram_max_ngram_size", 3 },
            { "engram_n_head_per_ngram", 5 }, { "engram_n_embed_per_ngram", 40 },
            { "engram_layer_head_vocab_sizes",
              { { "2", { { 11, 13, 17, 19, 23 }, { 29, 31, 37, 41, 43 } } } } },
        };
        Json config = {
            { "model_type", "hy_v4_flash" },
            { "architectures", { "HYV4FlashForConditionalGeneration" } },
            { "text_config", text }, { "dfly_num_hidden_layers", 1 },
            { "vision_config", { { "hidden_size", VE }, { "num_hidden_layers", 1 },
                                 { "intermediate_size", VF }, { "patch_size", VP },
                                 { "num_channels", 3 }, { "spatial_merge_size", VM },
                                 { "max_vit_seq_len", VPOS - 1 } } },
        };

        tiny::Rng r(4);
        std::vector<tiny::Tensor> ts;
        auto gain = [&](const std::string& n, int64_t k, const char* dt = "BF16") {
            ts.push_back(tiny::noise(n, { k }, r, 0.1f, 1.0f, dt));
        };
        auto mat = [&](const std::string& n, std::vector<int64_t> shape, const char* dt = "BF16") {
            ts.push_back(tiny::noise(n, std::move(shape), r, 0.2f, 0.0f, dt));
        };
        auto attn = [&](const std::string& p, int heads, const char* sink_dt) {
            mat(p + "q_a_proj.weight", { QL, H });
            gain(p + "q_a_layernorm.weight", QL);
            mat(p + "q_b_proj.weight", { heads * (NOPE + ROPE), QL });
            mat(p + "kv_a_proj_with_mqa.weight", { KVL + ROPE, H });
            gain(p + "kv_a_layernorm.weight", KVL);
            mat(p + "kv_b_proj.weight", { heads * (NOPE + VD), KVL });
            mat(p + "linear_gate.weight", { heads * VD, H });
            mat(p + "o_proj.weight", { H, heads * VD });
            gain(p + "post_proj_norm.weight", H);
            mat(p + "learnable_sink_param", { heads }, sink_dt);
        };
        auto mlp = [&](const std::string& p, int ff) {
            mat(p + "gate_proj.weight", { ff, H });
            mat(p + "up_proj.weight", { ff, H });
            mat(p + "down_proj.weight", { H, ff });
        };
        mat("model.embed_tokens.weight", { V, H });
        mat("lm_head.weight", { V, H });
        gain("model.norm.weight", H);
        mat("model.hc_head.hc_head_fn", { HC, HC * H }, "F32");
        mat("model.hc_head.hc_head_base", { HC }, "F32");
        mat("model.hc_head.hc_head_scale", { 1 }, "F32");
        for (int l = 0; l < 3; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            const bool sliding = l == 1;
            for (const char* hc : { "hc_attn_layer.hc_pre.", "hc_mlp_layer.hc_pre." }) {
                mat(p + hc + "hc_fn", { 2 * HC, HC * H }, "F32");
                mat(p + hc + "hc_base", { 2 * HC }, "F32");
                mat(p + hc + "hc_scale", { 2 }, "F32");
            }
            gain(p + "input_layernorm.weight", H);
            gain(p + "post_attention_layernorm.weight", H);
            attn(p + "self_attn.", sliding ? SWA_HEADS : HEADS, "F32");
            if (!sliding) {
                const std::string ix = p + "self_attn.compressed_indexer.";
                mat(ix + "query_projection.weight", { IXH * IXD, QL });
                mat(ix + "head_weight_projection.weight", { IXH, H });
                mat(ix + "token_key_projection.weight", { IXD, H });
                gain(ix + "token_key_norm.weight", IXD);
                mat(ix + "key_value_projection.weight", { IXD, IXD }, "F32");
                mat(ix + "key_gate_projection.weight", { IXD, IXD }, "F32");
                gain(ix + "key_norm.weight", IXD, "F32");
                mat(ix + "intra_group_ape", { POOL, IXD }, "F32");
            }
            if (l == 0) {
                mlp(p + "mlp.", FF);
                gain(p + "mlp.post_fc2_norm.weight", H);
            } else {
                mat(p + "mlp.gate.weight", { NE, H });
                mat(p + "mlp.gate.e_score_correction_bias", { NE }, "F32");
                mat(p + "mlp.routed_expert_down_proj.weight", { LAT, H });
                mat(p + "mlp.routed_expert_up_proj.weight", { H, LAT });
                mat(p + "mlp.experts.gate_up_proj", { NE, 2 * FE, LAT });
                mat(p + "mlp.experts.down_proj", { NE, LAT, FE });
                mlp(p + "mlp.shared_experts.", FE);
                gain(p + "mlp.combine_output_norm.weight", H);
            }
            if (l == 1) {
                const std::string e = p + "engram_layer.";
                mat(e + "multi_head_embedding.embedding.weight", { EROWS, EHD });
                tiny::Tensor map{ e + "lookup_table", "I32", { MAP }, {} };
                map.bytes.resize(MAP * 4);
                for (int32_t i = 0; i < MAP; ++i) {
                    const int32_t v = i / 2;
                    std::memcpy(map.bytes.data() + 4 * i, &v, 4);
                }
                ts.push_back(std::move(map));
                for (int h = 0; h < EHEADS; ++h)
                    gain(e + "lookup_per_head_norms." + std::to_string(h) + ".weight", EHD);
                for (int c = 0; c < HC; ++c) {
                    const std::string i = std::to_string(c);
                    mat(e + "key_projs." + i + ".weight", { H, EHEADS * EHD });
                    mat(e + "key_projs." + i + ".bias", { H });
                    gain(e + "norm_key." + i + ".weight", H);
                    gain(e + "norm_query." + i + ".weight", H);
                    mat(e + "unified_per_hc_scales." + i, { H });
                }
                mat(e + "value_proj.weight", { H, EHEADS * EHD });
                mat(e + "value_proj.bias", { H });
                gain(e + "unified_pre_gamma", H);
            }
        }
        {
            const std::string p = "model.dfly_layers.";
            gain(p + "0.input_layernorm.weight", H);
            gain(p + "0.post_attention_layernorm.weight", H);
            attn(p + "0.self_attn.", HEADS, "BF16");
            mlp(p + "0.mlp.", DFF);
            mat(p + "mtp_fusion.eh_proj.weight", { H, 2 * H });
            gain(p + "mtp_fusion.enorm.weight", H);
            gain(p + "mtp_fusion.hnorm.weight", H);
            mat(p + "mtp_fusion.gate", { H });
            gain(p + "hidden_correction.embed_norm.weight", H);
            gain(p + "hidden_correction.hidden_norm.weight", H);
            mat(p + "hidden_correction.gate_proj.weight", { H, 2 * H });
            mat(p + "hidden_correction.up_proj.weight", { H, 2 * H });
            mat(p + "hidden_correction.down_proj.weight", { H, H });
            gain(p + "final_norm.weight", H);
            mat(p + "confidence_head.proj.weight", { 1, H });
            mat(p + "confidence_head.proj.bias", { 1 });
        }
        {
            mat("vit.embeddings.patch_embedding.weight", { VE, 3, VP, VP });
            mat("vit.embeddings.patch_embedding.bias", { VE });
            mat("vit.embeddings.position_embedding.weight", { VPOS, VE });
            const std::string p = "vit.layers.0.";
            for (const char* n : { "input_layernorm", "post_attention_layernorm" }) {
                gain(p + n + ".weight", VE);
                mat(p + n + ".bias", { VE });
            }
            for (const char* n : { "q_proj", "k_proj", "v_proj", "o_proj" }) {
                mat(p + "self_attn." + n + ".weight", { VE, VE });
                mat(p + "self_attn." + n + ".bias", { VE });
            }
            mat(p + "mlp.dense_h_to_4h.weight", { VF, VE });
            mat(p + "mlp.dense_h_to_4h.bias", { VF });
            mat(p + "mlp.dense_4h_to_h.weight", { VE, VF });
            mat(p + "mlp.dense_4h_to_h.bias", { VE });
            gain("vit.perceive.norm.weight", VE);
            mat("vit.perceive.proj.0.weight", { VE * VM * VM, VE, VM, VM });
            mat("vit.perceive.proj.0.bias", { VE * VM * VM });
            mat("vit.perceive.proj.2.weight", { H, VE * VM * VM, 1, 1 });
            mat("vit.perceive.proj.2.bias", { H });
            mat("vit.perceive.image_newline", { H });
        }
        if (!tiny::write_checkpoint(ckpt, config, ts)) return;
        for (const auto& t : ts) src[t.name] = std::string(t.bytes.begin(), t.bytes.end());
        ok = true;
    }

    int convert(const std::string& out, std::vector<std::string> more, std::string* log) const {
        std::vector<std::string> argv{ RAD_E2E_CONVERT, ckpt, "-o", out, "--max-ctx", "64",
                                       "--max-tok", "16" };
        argv.insert(argv.end(), more.begin(), more.end());
        const int st = tiny::run(argv, { std::string("RADIANCE_HOME=") + RAD_E2E_HOME,
                                         "ROCR_VISIBLE_DEVICES=99", "OMP_NUM_THREADS=2",
                                         "OMP_WAIT_POLICY=passive" }, log);
        if (st) fprintf(stderr, "%s\n", log->c_str());
        return st;
    }
};

const Fixture& fixture() {
    static const Fixture f;
    return f;
}

/* The bytes of `name`'s only plane, or "" (and a failed check) where it has none or several. */
std::string plain(const RadFile& f, const std::string& name) {
    const RadFileEntry* e = f.find(name);
    if (!e || e->n_planes != 1) {
        fprintf(stderr, "'%s': %s\n", name.c_str(), e ? "more than one plane" : "no such entry");
        CHECK(false);
        return {};
    }
    const RadFilePlane& p = f.plane(*e, 0);
    return std::string((const char*)f.data(p), (size_t)p.bytes);
}
}  // namespace

TEST(every_checkpoint_tensor_is_read_and_kept_as_the_release_holds_it) {
    const Fixture& fx = fixture();
    REQUIRE(fx.ok);
    std::string log;
    const std::string out = fx.dir / "plain.rad";
    REQUIRE_EQ(fx.convert(out, {}, &log), 0);
    /* rad-convert names the tensors no declared weight reads; the map leaves none. */
    CHECK(log.find("are read by no declared weight") == std::string::npos);
    RadFile f;
    REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
    CHECK_EQ(std::string(f.str(f.header().arch_id)), std::string("hyv4flash"));
    auto at = [&](const std::string& n) { return fx.src.at(n); };

    CHECK(plain(f, "token_embd.weight") == at("model.embed_tokens.weight"));
    CHECK(plain(f, "output.weight") == at("lm_head.weight"));
    CHECK(plain(f, "hc_head.fn") == at("model.hc_head.hc_head_fn"));
    const std::string l0 = "model.layers.0.", l1 = "model.layers.1.";
    CHECK(plain(f, "blk.0.hc_attn.fn") == at(l0 + "hc_attn_layer.hc_pre.hc_fn"));
    CHECK(plain(f, "blk.0.hc_ffn.scale") == at(l0 + "hc_mlp_layer.hc_pre.hc_scale"));
    CHECK(plain(f, "blk.0.attn_kv_a.weight") == at(l0 + "self_attn.kv_a_proj_with_mqa.weight"));
    CHECK(plain(f, "blk.0.attn_gate.weight") == at(l0 + "self_attn.linear_gate.weight"));
    CHECK(plain(f, "blk.0.attn_sinks.weight") == at(l0 + "self_attn.learnable_sink_param"));
    CHECK(plain(f, "blk.0.index_pool_gate.weight") ==
          at(l0 + "self_attn.compressed_indexer.key_gate_projection.weight"));
    CHECK(plain(f, "blk.0.index_ape") == at(l0 + "self_attn.compressed_indexer.intra_group_ape"));
    /* The sliding layer has the wider head count and no indexer. */
    CHECK_EQ(plain(f, "blk.1.attn_q_b.weight").size(), (size_t)SWA_HEADS * (NOPE + ROPE) * QL * 2);
    CHECK(f.find("blk.1.index_query.weight") == nullptr);
    /* gate and up are fused, gate rows first, in the dense layer and in the shared expert. */
    CHECK(plain(f, "blk.0.ffn_gate_up.weight") ==
          at(l0 + "mlp.gate_proj.weight") + at(l0 + "mlp.up_proj.weight"));
    CHECK(plain(f, "blk.0.ffn_post_norm.weight") == at(l0 + "mlp.post_fc2_norm.weight"));
    CHECK(plain(f, "blk.1.ffn_gate_up_shexp.weight") ==
          at(l1 + "mlp.shared_experts.gate_proj.weight") +
              at(l1 + "mlp.shared_experts.up_proj.weight"));
    CHECK(plain(f, "blk.1.ffn_post_norm.weight") == at(l1 + "mlp.combine_output_norm.weight"));
    /* The latent's in and out are the checkpoint's "down" and "up", by direction in width. */
    CHECK(plain(f, "blk.1.ffn_latent_in.weight") == at(l1 + "mlp.routed_expert_down_proj.weight"));
    CHECK(plain(f, "blk.1.ffn_latent_out.weight") == at(l1 + "mlp.routed_expert_up_proj.weight"));
    CHECK(plain(f, "blk.1.ffn_gate_inp.correction") ==
          at(l1 + "mlp.gate.e_score_correction_bias"));
    /* Each expert is its dim-0 slice of the two stacked tensors. */
    const std::string& gu = at(l1 + "mlp.experts.gate_up_proj");
    const std::string& dn = at(l1 + "mlp.experts.down_proj");
    const size_t gu_n = (size_t)2 * FE * LAT * 2, dn_n = (size_t)LAT * FE * 2;
    for (int e = 0; e < NE; ++e) {
        const std::string i = std::to_string(e);
        CHECK(plain(f, "blk.1.ffn_gate_up_exps." + i + ".weight") == gu.substr(e * gu_n, gu_n));
        CHECK(plain(f, "blk.1.ffn_down_exps." + i + ".weight") == dn.substr(e * dn_n, dn_n));
    }
    /* The Engram: the table and the token map as shipped, the per-stream and per-head families
     * joined in index order (ten heads: two maps of the name). */
    const std::string eg = l1 + "engram_layer.";
    CHECK(plain(f, "blk.1.engram_table.weight") ==
          at(eg + "multi_head_embedding.embedding.weight"));
    CHECK(plain(f, "blk.1.engram_token_map") == at(eg + "lookup_table"));
    std::string heads, keys, branch;
    for (int h = 0; h < EHEADS; ++h)
        heads += at(eg + "lookup_per_head_norms." + std::to_string(h) + ".weight");
    for (int c = 0; c < HC; ++c) {
        keys += at(eg + "key_projs." + std::to_string(c) + ".weight");
        branch += at(eg + "unified_per_hc_scales." + std::to_string(c));
    }
    CHECK(plain(f, "blk.1.engram_head_norm.weight") == heads);
    CHECK(plain(f, "blk.1.engram_key.weight") == keys);
    CHECK(plain(f, "blk.1.engram_branch_scale.weight") == branch);
    CHECK(f.find("blk.0.engram_table.weight") == nullptr);
    /* The hash constants are not in the checkpoint; without --add they are left out. */
    CHECK(f.find("blk.1.engram_mult") == nullptr);
    /* The drafter's MLP width is its tensor's, not three times the hidden size. */
    CHECK(plain(f, "dfly.blk.0.ffn_gate_up.weight") ==
          at("model.dfly_layers.0.mlp.gate_proj.weight") +
              at("model.dfly_layers.0.mlp.up_proj.weight"));
    CHECK(plain(f, "dfly.fusion.weight") == at("model.dfly_layers.mtp_fusion.eh_proj.weight"));
    CHECK(plain(f, "v.patch.weight") == at("vit.embeddings.patch_embedding.weight"));
    CHECK(plain(f, "v.merger.newline") == at("vit.perceive.image_newline"));
}

TEST(max_spec_0_leaves_the_drafter_out) {
    const Fixture& fx = fixture();
    REQUIRE(fx.ok);
    std::string log;
    const std::string out = fx.dir / "nodraft.rad";
    REQUIRE_EQ(fx.convert(out, { "--max-spec", "0" }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
    CHECK(f.find("dfly.blk.0.attn_q_a.weight") == nullptr);
    CHECK(f.find("blk.2.attn_q_a.weight") != nullptr);
}

TEST(the_bf16_recipe_stores_the_engram_table_as_e4m3_rows) {
    const Fixture& fx = fixture();
    REQUIRE(fx.ok);
    std::string log;
    const std::string out = fx.dir / "bf16.rad";
    REQUIRE_EQ(fx.convert(out, { "--recipe", RAD_E2E_RECIPES "/hy4-flash-bf16.recipe" }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
    const RadFileEntry* e = f.find("blk.1.engram_table.weight");
    REQUIRE(e != nullptr);
    /* Codes and one scale a row. */
    REQUIRE_EQ(e->n_planes, (uint32_t)2);
    const RadEncoding& enc = f.encoding(*e);
    CHECK_EQ(enc.plane[0].dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(f.plane(*e, 0).bytes, (uint64_t)EROWS * EHD);
    CHECK_EQ(enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(f.plane(*e, 1).bytes, (uint64_t)EROWS * 2);
    /* Every other weight is the checkpoint's. */
    CHECK(plain(f, "blk.1.ffn_gate_up_exps.0.weight").size() == (size_t)2 * FE * LAT * 2);
}

TEST(the_w4nl64_recipe_stores_every_routed_expert_as_four_bit_codes) {
    const Fixture& fx = fixture();
    REQUIRE(fx.ok);
    std::string log;
    const std::string out = fx.dir / "w4.rad";
    REQUIRE_EQ(fx.convert(out, { "--recipe", RAD_E2E_RECIPES "/hy4-flash-w4nl64-rtn.recipe" },
                          &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
    for (int l = 1; l < 3; ++l)
        for (int e = 0; e < NE; ++e)
            for (const char* proj : { "ffn_gate_up_exps", "ffn_down_exps" }) {
                const std::string n = "blk." + std::to_string(l) + "." + proj + "." +
                                      std::to_string(e) + ".weight";
                const RadFileEntry* en = f.find(n);
                REQUIRE(en != nullptr);
                const RadEncoding& enc = f.encoding(*en);
                CHECK_EQ(enc.plane[0].dtype, (uint32_t)RAD_U4);
                CHECK(rad_enc_plane(&enc, "table") != nullptr);
            }
    /* The shared expert and the latent's projections are not routed experts. */
    CHECK_EQ(f.find("blk.1.ffn_gate_up_shexp.weight")->n_planes, (uint32_t)1);
    CHECK_EQ(f.find("blk.1.ffn_latent_in.weight")->n_planes, (uint32_t)1);
    CHECK_EQ(f.find("blk.1.engram_table.weight")->n_planes, (uint32_t)2);
}

TEST(the_hash_constants_come_from_an_added_file) {
    const Fixture& fx = fixture();
    REQUIRE(fx.ok);
    /* What data/hy4flash_engram_tables.py writes for the one Engram layer: I64, by layer index. */
    auto i64 = [](const std::string& name, const std::vector<int64_t>& v) {
        tiny::Tensor t{ name, "I64", { (int64_t)v.size() }, {} };
        t.bytes.resize(v.size() * 8);
        std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
        return t;
    };
    const std::vector<int64_t> sizes{ 11, 13, 17, 19, 23, 29, 31, 37, 41, 43 };
    std::vector<int64_t> offsets;
    for (int64_t at = 0; int64_t n : sizes) { offsets.push_back(at); at += n; }
    const std::vector<tiny::Tensor> ts{ i64("engram.1.hash_multipliers", { 7, 0x4000000000001ll, 5 }),
                                        i64("engram.1.head_vocab_sizes", sizes),
                                        i64("engram.1.head_offsets", offsets) };
    const std::string add = fx.dir / "engram-tables.safetensors";
    REQUIRE(tiny::write_safetensors(add, ts));
    std::string log;
    const std::string out = fx.dir / "hash.rad";
    REQUIRE_EQ(fx.convert(out, { "--add", add }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
    auto bytes = [](const tiny::Tensor& t) { return std::string(t.bytes.begin(), t.bytes.end()); };
    CHECK(plain(f, "blk.1.engram_mult") == bytes(ts[0]));
    CHECK(plain(f, "blk.1.engram_vocab_sizes") == bytes(ts[1]));
    CHECK(plain(f, "blk.1.engram_offsets") == bytes(ts[2]));
}

/* The release's tokenizer and chat template are baked, with the special tokens the template and the
 * reply format name. The release's files are 9 MB, so they are read from the directory
 * RAD_TEST_HY4_TOKENIZER_DIR names (tokenizer.json, chat_template.jinja) and the case is skipped
 * without it. */
TEST(the_release_tokenizer_and_chat_template_are_baked) {
    const char* dir = std::getenv("RAD_TEST_HY4_TOKENIZER_DIR");
    if (!dir || !*dir) SKIP("set RAD_TEST_HY4_TOKENIZER_DIR to a directory with the release's tokenizer.json and chat_template.jinja");
    const Fixture& fx = fixture();
    REQUIRE(fx.ok);
    const std::string tj = std::string(dir) + "/tokenizer.json", ct = std::string(dir) + "/chat_template.jinja";
    std::string log;
    const std::string out = fx.dir / "tok.rad";
    REQUIRE_EQ(fx.convert(out, { "--tokenizer", tj, "--chat-template", ct }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
    VocabBuild vb;
    REQUIRE_EQ(vocab_deserialize(f.base(), (uint64_t)f.size(), f.vocab().h, vb), RAD_OK);
    std::ifstream in(ct, std::ios::binary);
    std::ostringstream want;
    want << in.rdbuf();
    std::string tmpl = want.str();
    while (!tmpl.empty() && (tmpl.back() == '\n' || tmpl.back() == ' ')) tmpl.pop_back();
    CHECK_EQ(vb.chat_template, tmpl);
    CHECK_EQ((int)vb.tokens.size(), 120832);
    /* The ids the template and the reply format rely on, spelled with the release's suffix. */
    const std::map<int, std::string> expect{
        { 120000, "<｜hy_start:opensource｜>" },   { 120001, "<｜hy_middle:opensource｜>" },
        { 120025, "<｜hy_end:opensource｜>" },     { 120029, "<think:opensource>" },
        { 120030, "</think:opensource>" },         { 120033, "<tool_call:opensource>" },
        { 120034, "</tool_call:opensource>" },     { 120035, "<tool_calls:opensource>" },
        { 120036, "</tool_calls:opensource>" },    { 120037, "<tool_response:opensource>" },
        { 120039, "<｜reasoning_mode:opensource｜>" }, { 120040, "<arg_key:opensource>" },
        { 120041, "</arg_key:opensource>" },       { 120042, "<arg_value:opensource>" },
        { 120043, "</arg_value:opensource>" } };
    for (const auto& [id, text] : expect) CHECK_EQ(vb.tokens.at((size_t)id), text);
    CHECK_EQ(vb.bos, 120000);
    CHECK_EQ(vb.eos, 120025);
}

RAD_TEST_MAIN()
