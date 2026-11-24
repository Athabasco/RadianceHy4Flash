/* table_fill_test.cpp -- the Engine's deferred Engram fill (B-01).
 *
 * copy_table reserves and publishes a table's memory as the load's walk meets its weight, starts
 * the read, and join_table_fills waits for every table it started once the container pass has run
 * beside them. These tests drive the same reserve/start/join/release path a --ngram-placement ram
 * server drives for the tables it reads for the cards, on the host backend's memory: the locked
 * copy (a host gather) and the pinned one (a card's gather), without a card.
 */
#include "rad_test.h"
#include "engine_priv.h"
#include "format/radfile.h"
#include "format/table_copy.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rad {
/* The walk, copy_table and join_table_fills are private, as the other bringup tests reach them. */
struct TableFillTestAccess {
    static void hold(Engine& e, std::unique_ptr<RadFile> f) { e.file_ = std::move(f); }
    static int  copy(Engine& e, const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                     std::string* why) {
        return e.copy_table(src, n, foff, card, out, why);
    }
    static int    join(Engine& e) { return e.join_table_fills(); }
    static size_t pending(Engine& e) { return e.pending_tables_.size(); }
    static size_t locked(Engine& e) { return e.table_copies_.size(); }
    static size_t pinned(Engine& e) { return e.card_copies_.size(); }
};
}  // namespace rad

namespace {

/* Bytes no 16-byte shift of which can look like another: the fill's check samples them. */
std::vector<uint8_t> pattern_bytes(int64_t n, uint32_t seed) {
    std::vector<uint8_t> b((size_t)n);
    uint32_t x = seed | 1u;
    for (size_t i = 0; i < b.size(); ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        b[i] = (uint8_t)(x >> 11);
    }
    return b;
}

/* A container with three n-gram tables of different sizes, and the mapping the copies read from.
 * c comes first, so the 16-byte shift the failure case reads stays inside the container. */
struct Tables {
    std::string                   path;
    std::vector<uint8_t>          want[3];
    const uint8_t*                src[3] = {nullptr, nullptr, nullptr};
    int64_t                       off[3] = {0, 0, 0};
    int64_t                       n[3]   = {0, 0, 0};
    std::unique_ptr<rad::RadFile> file;

    int open() {
        file = std::make_unique<rad::RadFile>();
        if (file->open(path.c_str()) < 0) return -1;
        const char* names[] = {"ngram.c", "ngram.a", "ngram.b"};
        for (int i = 0; i < 3; ++i) {
            const RadFileEntry* e = file->find(names[i]);
            if (!e) return -1;
            const RadFilePlane& p = file->plane(*e, 0);
            src[i] = file->data(p);
            off[i] = (int64_t)p.offset;
            n[i] = (int64_t)p.bytes;
        }
        return 0;
    }
};

struct TempContainer {
    std::string path;
    ~TempContainer() { std::remove(path.c_str()); }
};

/* Writes the three tables; false when the writer refused. */
bool write_tables(const std::string& path, int64_t nc, int64_t na, int64_t nb, Tables* t) {
    t->path = path;
    t->n[0] = nc;
    t->n[1] = na;
    t->n[2] = nb;
    t->want[0] = pattern_bytes(nc, 0xc2b2ae35u);
    t->want[1] = pattern_bytes(na, 0x9e3779b9u);
    t->want[2] = pattern_bytes(nb, 0x85ebca6bu);
    rad::RadWriter writer;
    if (writer.begin(path.c_str()) < 0) return false;
    writer.set_arch("radtest");
    const char* names[] = {"ngram.c", "ngram.a", "ngram.b"};
    for (int i = 0; i < 3; ++i) {
        rad::RadWriter::Weight w;
        w.name = names[i];
        w.enc = rad_enc_plain(RAD_BF16);
        w.rank = 2;
        w.shape[0] = t->n[i] / 2;   /* bf16: two bytes a row */
        w.shape[1] = 1;
        const void* planes[] = {t->want[i].data()};
        if (writer.add_weight(w, planes) < 0) return false;
    }
    return writer.finish() == RAD_OK;
}

bool same_as(const void* got, const uint8_t* want, int64_t n) {
    return got && std::memcmp(got, want, (size_t)n) == 0;
}

}  // namespace

/* The locked copy (a host gather reads it): two tables started before either is waited on, one
 * copy for the same table a second rank asks for, publication only at the join, and a fill that
 * read the wrong bytes released and failing the load there. */
TEST(a_host_gathers_tables_read_before_the_join_and_are_published_after_it) {
    TempContainer tmp{radtest::tmp_path("table-fill-locked.rad")};
    const int64_t nc = 64ll << 10, na = 3ll << 20, nb = 1ll << 20;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    rad::Config config;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-locked-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    const int rc = rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], false, &outa, &why);
    /* A locked-memory limit below the tables is the machine's setting, not a defect. */
    if (rc == RAD_E_NOMEM && why.find("mlock2") != std::string::npos) SKIP("%s", why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], false, &outb, &why),
               RAD_OK);
    CHECK(outa != nullptr && outa != (const void*)src[1]);
    CHECK(outb != nullptr && outb != (const void*)src[2]);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    /* Nothing is published before its bytes are in: the join is what publishes. */
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));

    /* The same table for another rank is the one copy, not a second read. */
    void* again = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], false, &again, &why),
               RAD_OK);
    CHECK(again == outa);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(2));
    /* Byte for byte, not the fill's samples: the tables are the container's bytes. */
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));

    /* A fill that read the wrong bytes is not published: the join fails the load and frees it. */
    void* bad = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[0], n[0], off[0] + 16, false, &bad, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK_EQ(rad::TableFillTestAccess::join(engine), (int)RAD_E_IO);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(2));   /* the failed one is not */
    /* Joined and cleared: a second join is nothing left to do. */
    CHECK_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
}

/* The pinned copy (a card reads it over the link): the same deferral, with the copies published
 * to the engine's pinned ones only at the join. */
TEST(a_cards_tables_read_before_the_join_and_are_published_after_it) {
    TempContainer tmp{radtest::tmp_path("table-fill-pinned.rad")};
    const int64_t nc = 64ll << 10, na = 3ll << 20, nb = 1ll << 20;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    rad::Config config;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-pinned-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], true, &outa, &why),
               RAD_OK);
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], true, &outb, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(0));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));
}

RAD_TEST_MAIN()
