/* r4d_fuse_test.cpp -- the chain a fused row publishes (`RadKernelInfo::replaces`,
 * libr4d/r4d_fuse.cpp): which conventional ops it names, and the operand wiring it hands each of
 * them, in that op's schema order.
 *
 * A row that publishes no chain is counted uncovered by rad-kbench's fusion oracle and produces no
 * substitution advisory in --debug-graph, so the chain is what makes a fused kernel's claim to be
 * byte-identical to its unfused sequence checkable at all. The chains are hooks rather than tables
 * (abi/rad_abi.h), so their wiring can only be checked by calling them; here that is done on the
 * host, at the geometry the fused row itself declares.
 *
 * K-15's row (mla_attn_paged_sink_expand_i8g) is the case this file was opened for: the sink's
 * launch, `out` = the latent, then the expansion over that latent and the row's own operands.
 *
 * libr4d's hooks are host C++, compiled into this test (tests/CMakeLists.txt).
 */
#include "rad_test.h"
#include "r4d_plugin.h"

#include <cstring>
#include <vector>

namespace {

/* The fused row's geometry as r4d_rows.cpp declares it (pMlaAttnPagedSinkX): the decode instance
 * the row is built for, one shape of many. */
std::vector<RadParam> k15_geometry() {
    return { RAD_INT("M", 16),
             RAD_INT("heads", 32),
             RAD_INT("latent_dim", 512),
             RAD_INT("rope_dim", 64),
             RAD_INT("block_size", 16),
             RAD_INT("window", 256),
             RAD_F64("scale", 0.041666666666666664),
             RAD_STR("dtype", "bf16"),
             RAD_INT("stable", 1),
             RAD_INT("k_dim", 192),
             RAD_INT("v_dim", 256),
             RAD_INT("group", 128) };
}

const RadParam* param(const RadFuseStep& s, const char* k) {
    for (int i = 0; i < s.n_p; ++i)
        if (s.p[i].key && std::strcmp(s.p[i].key, k) == 0) return &s.p[i];
    return nullptr;
}

const RadParam* param(const std::vector<RadParam>& g, const char* k) {
    for (const RadParam& p : g)
        if (p.key && std::strcmp(p.key, k) == 0) return &p;
    return nullptr;
}

/* A step's parameter is the fused op's own, not the default the step op would take: a chain run at
 * the wrong geometry produces a difference that reads as a kernel fault. */
void check_mirrored(const RadFuseStep& s, const std::vector<RadParam>& g, const char* k) {
    const RadParam* want = param(g, k);
    const RadParam* got = param(s, k);
    CHECK(want != nullptr);
    CHECK(got != nullptr);
    if (!want || !got) return;
    CHECK_EQ(got->kind, want->kind);
    CHECK(got->kind != RAD_P_STR || std::strcmp(got->sval, want->sval) == 0);
    CHECK_EQ(got->ival, want->ival);
    CHECK_EQ(got->ihi, want->ihi);
    CHECK(got->dval == want->dval);
}

/* The chain as a caller walks it (tools/rad_kbench.cpp): RAD_E_NOTFOUND ends it. A refused step
 * (`RAD_E_UNSUPPORTED`, no chain at this geometry) ends it too; the tests that need the chain
 * check its size, and the refusal test asks the hook directly. */
std::vector<RadFuseStep> chain(const std::vector<RadParam>& g, uint64_t present) {
    std::vector<RadFuseStep> steps;
    for (int step = 0; step < 8; ++step) {
        RadFuseStep fs{};
        const int rc = r4d_fuse_mla_attn_paged_sink_expand_i8g(g.data(), (int)g.size(), present,
                                                               step, &fs);
        if (rc != RAD_OK) break;
        steps.push_back(fs);
    }
    return steps;
}

}  /* namespace */

/* The sink over operands 0-7 (its own row replaces its split with merge_sink, which is what leaves
 * `out` the latent), then the expansion over that latent as `x`. */
TEST(k15_chain_is_the_sink_and_the_expansion_over_its_latent) {
    const std::vector<RadParam> g = k15_geometry();
    const std::vector<RadFuseStep> s = chain(g, RAD_FUSE_PRESENT_ALL);
    CHECK_EQ(s.size(), 2u);
    if (s.size() != 2) return;

    CHECK(std::strcmp(s[0].op, "mla_attn_paged_sink") == 0);
    CHECK_EQ(s[0].n_from, 8);                      /* q, q_rope, kv, block_table, seqused,
                                                    * cu_seqlens, sinks, out */
    for (int i = 0; i < s[0].n_from && i < 8; ++i) CHECK_EQ(s[0].from[i], i);
    CHECK_EQ(s[0].n_p, 9);
    for (const char* k : { "M", "heads", "latent_dim", "rope_dim", "block_size", "window", "scale",
                           "dtype", "stable" })
        check_mirrored(s[0], g, k);

    CHECK(std::strcmp(s[1].op, "mla_expand_gate_had_quant_i8g") == 0);
    CHECK_EQ(s[1].n_from, 6);                      /* x, w, g, q, scale, out_bf16 */
    const int16_t want[6] = { 7, 8, 9, 10, 11, 12 };
    for (int i = 0; i < s[1].n_from && i < 6; ++i) CHECK_EQ(s[1].from[i], want[i]);
    CHECK_EQ(s[1].n_p, 7);
    for (const char* k : { "M", "heads", "k_dim", "v_dim", "latent_dim", "group", "dtype" })
        check_mirrored(s[1], g, k);
}

/* Without `out_bf16` the fused kernel keeps the gated product in a temporary, and the expansion's
 * row does the same with `out_bf16` absent, so the step passes no operand there. */
TEST(k15_chain_leaves_out_bf16_unwired_when_the_row_does_not_take_it) {
    const std::vector<RadParam> g = k15_geometry();
    const std::vector<RadFuseStep> s = chain(g, RAD_FUSE_PRESENT_ALL & ~(1ull << 12));
    CHECK_EQ(s.size(), 2u);
    if (s.size() != 2) return;
    CHECK_EQ(s[1].n_from, 6);
    if (s[1].n_from != 6) return;
    CHECK_EQ(s[1].from[5], (int16_t)RAD_FUSE_NONE);
}

/* A geometry the expansion's own row does not serve, and one the sink's cannot be resolved at, have
 * no chain: a step resolved there would be a different op's geometry. */
TEST(k15_chain_refuses_a_geometry_it_cannot_expand) {
    RadFuseStep fs{};

    std::vector<RadParam> g = k15_geometry();
    for (RadParam& p : g)
        if (p.key && std::strcmp(p.key, "group") == 0) p.ival = 64;
    CHECK(r4d_fuse_mla_attn_paged_sink_expand_i8g(g.data(), (int)g.size(), RAD_FUSE_PRESENT_ALL, 0,
                                                  &fs) < 0);
    CHECK(chain(g, RAD_FUSE_PRESENT_ALL).empty());

    g = k15_geometry();
    g.erase(g.begin() + 5);                        /* no `window`: the sink's own geometry */
    CHECK(r4d_fuse_mla_attn_paged_sink_expand_i8g(g.data(), (int)g.size(), RAD_FUSE_PRESENT_ALL, 0,
                                                  &fs) < 0);
    CHECK(chain(g, RAD_FUSE_PRESENT_ALL).empty());

    g = k15_geometry();
    CHECK(r4d_fuse_mla_attn_paged_sink_expand_i8g(g.data(), (int)g.size(), RAD_FUSE_PRESENT_ALL, 0,
                                                  nullptr) == RAD_E_INVAL);
}

RAD_TEST_MAIN()
