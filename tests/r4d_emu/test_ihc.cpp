/* tests/r4d_emu: K-12's warm of the junction row's operands on the host -- r4d_ihc.h's warm_row,
 * which the all-reduce twin's block runs while thread 0 waits on the peer's flag
 * (r4d_ar_ihc_junction.hip). That branch needs a peer rank, which the emulator has not, so what is
 * held here is the warm itself, the whole of the change: it reads the row's own operands, it reads
 * each of them to its last word, and it reads nothing else.
 *
 * The operands sit against inaccessible pages -- every row's bytes end at the last byte before a
 * PROT_NONE page, and a row that is whole pages also opens with one -- so a read past a row's end,
 * into a pitched row's padding or before a row's start faults, and the operands' own pages are
 * PROT_READ, so a write to one faults too. A JuncRow built here carries no operand the warm has no
 * business with (post_row, scale, base, cq, ... are null), so a warm that touched one of those
 * faults as well. A row without a tail carries no fn and no next gain (null), so a warm that read
 * them faults.
 *
 * Three things are asserted of every geometry: with the operands readable the warm completes and
 * leaves every byte as it was (loads only); with one operand's last page taken away it faults
 * (the warm reaches the end of every operand it is given, or the lever is quietly a fraction of
 * itself); and thread 0, the one waiting, reads nothing even when every operand is inaccessible.
 * The cases that must fault run under a SIGSEGV handler; one that must not fault reports the
 * fault as its failure. */
#include "r4d_ihc.h"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <setjmp.h>
#include <sys/mman.h>
#include <vector>

static const size_t kPage = 4096;

/* An operand: `rows` rows of `bytes` at a pitch a page wider than a row needs, each row's bytes
 * ending at the last byte before an inaccessible page. `dead_tail` takes every row's last page
 * away as well, for the cases that must fault where a row ends. */
struct Rows {
    char* map = nullptr;
    size_t map_len = 0, lead = 0, span = 0, stride = 0, bytes = 0;
    int rows = 0;
    std::vector<char> keep;

    ~Rows() { if (map) munmap(map, map_len); }

    bool make(int nrows, size_t nbytes, bool dead_tail) {
        rows = nrows;
        bytes = nbytes;
        span = (bytes + kPage - 1) / kPage * kPage;
        lead = bytes % kPage == 0 ? kPage : 0;   /* a page before a row that is whole pages */
        stride = lead + span + kPage;            /* and one after it */
        map_len = (size_t)rows * stride;
        map = (char*)mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map == MAP_FAILED) { map = nullptr; return false; }
        keep.resize((size_t)rows * bytes);
        for (int j = 0; j < rows; ++j) {
            char* p = row(j);
            for (size_t i = 0; i < bytes; ++i)
                keep[(size_t)j * bytes + i] = p[i] = (char)(i * 37u + 11u + 5u * (unsigned)j);
            char* base = map + (size_t)j * stride;
            if (mprotect(base + lead, span, PROT_READ) != 0) return false;   /* the row's own pages */
            if (mprotect(base + lead + span, kPage, PROT_NONE) != 0) return false;   /* past its end */
            if (lead && mprotect(base, lead, PROT_NONE) != 0) return false;          /* before it */
            if (dead_tail && mprotect(base + lead + span - kPage, kPage, PROT_NONE) != 0) return false;
        }
        return true;
    }
    char* row(int j) const { return map + (size_t)j * stride + lead + (span - bytes); }
    uint64_t pitch() const { return stride / 4; }   /* in f32 elements, as JuncRow::fp is */
    bool intact() const {
        for (int j = 0; j < rows; ++j)
            if (std::memcmp(row(j), keep.data() + (size_t)j * bytes, bytes) != 0) return false;
        return true;
    }
};

static sigjmp_buf g_jmp;
static volatile sig_atomic_t g_sig = 0;
static void on_fault(int sig) { g_sig = sig; siglongjmp(g_jmp, 1); }

/* Runs f with a fault caught: the signal, or 0 when f returned. */
template <class F>
static int faults(F f) {
    struct sigaction sa {}, old_segv {}, old_bus {};
    sa.sa_handler = on_fault;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, &old_segv) != 0 || sigaction(SIGBUS, &sa, &old_bus) != 0) return -1;
    g_sig = 0;
    if (sigsetjmp(g_jmp, 1) == 0) f();
    int sig = (int)g_sig;
    sigaction(SIGSEGV, &old_segv, nullptr);
    sigaction(SIGBUS, &old_bus, nullptr);
    return sig;
}

/* The row as the all-reduce twin builds it. Everything the warm must not read stays null. */
template <int HC, bool POST, bool TAIL>
struct Row {
    Rows h, wp, wn, fn;
    r4d_ihc::JuncRow r{};
    bool make(uint32_t n, int dead) {
        constexpr int R = POST ? 2 * HC : HC;
        const size_t hn = (size_t)HC * n;   /* the row's streams, one row of fn's values */
        r.n = n;
        if (!h.make(1, hn * 2, (dead & 1) != 0)) return false;
        r.hr = (uint16_t*)h.row(0);
        if (!wp.make(1, (size_t)n * 4, (dead & 2) != 0)) return false;
        r.w_post = (const float*)wp.row(0);
        if (TAIL) {
            if (!wn.make(1, (size_t)n * 4, (dead & 4) != 0)) return false;
            r.w_next = (const float*)wn.row(0);
            if (!fn.make(R, hn * 4, (dead & 8) != 0)) return false;
            r.fn = (const float*)fn.row(0);
            r.fp = fn.pitch();
        }
        return true;
    }
};

/* Threads 1..kThreads - 1, as the block's else branch runs them; wave 0's lanes 1..31 are one
 * wave with the spinning thread 0 and so run theirs after it, the other fifteen waves during it. */
template <int HC, bool POST, bool TAIL>
static void warm_block(const r4d_ihc::JuncRow& r) {
    for (uint32_t tid = 1; tid < (uint32_t)r4d_ihc::kThreads; ++tid) r4d_ihc::warm_row<HC, POST, TAIL>(r, tid);
}

template <int HC, bool POST, bool TAIL>
static void one(uint32_t n, int& cases, int& fails) {
    auto fail = [&](const char* why) {
        ++fails;
        std::printf("FAIL warm_row hc %d n %u post %d tail %d: %s\n", HC, n, (int)POST, (int)TAIL, why);
    };
    const int nops = TAIL ? 4 : 2;
    static const char* const opname[4] = {"h", "w_post", "w_next", "fn"};

    ++cases;
    {
        Row<HC, POST, TAIL> b;
        if (!b.make(n, 0)) { fail("the operands' mapping"); return; }
        int sig = faults([&] { warm_block<HC, POST, TAIL>(b.r); });
        if (sig) {
            fail("a fault inside an operand: the warm reads past a row's end, into a pitched row's "
                 "padding or before it, or reads an operand this row has not");
            return;
        }
        bool kept = b.h.intact() && b.wp.intact();
        if (TAIL) kept = kept && b.wn.intact() && b.fn.intact();
        if (!kept) fail("the warm wrote an operand: it must be loads only");
    }
    /* every operand is read to its last page: with that page gone the warm must fault there */
    for (int op = 0; op < nops; ++op) {
        ++cases;
        Row<HC, POST, TAIL> c;
        if (!c.make(n, 1 << op)) { fail("the operands' mapping"); continue; }
        if (!faults([&] { warm_block<HC, POST, TAIL>(c.r); })) {
            static char why[128];
            std::snprintf(why, sizeof why, "the warm stopped short of %s's last page", opname[op]);
            fail(why);
        }
    }
    /* thread 0 is the one waiting on the peer's flag */
    ++cases;
    {
        Row<HC, POST, TAIL> c;
        if (!c.make(n, 0xF)) fail("the operands' mapping");
        else if (faults([&] { r4d_ihc::warm_row<HC, POST, TAIL>(c.r, 0u); }))
            fail("thread 0 read an operand: it is the thread that waits, not a warmer");
    }
}

int main() {
    int cases = 0, fails = 0;
    /* the model's decode row: hc 4, n 3072, a tail's fn of eight rows of 12288 f32 (393 KB) */
    one<4, true, true>(3072, cases, fails);
    /* the same row at a verify, without a tail: no fn and no next gain to read */
    one<4, true, false>(3072, cases, fails);
    /* no post, so fn is two rows, and a row far narrower than the block of threads */
    one<2, false, true>(40, cases, fails);
    /* rows of a few hundred bytes, so a row shares its first page with the mapping's lead */
    one<3, true, true>(24, cases, fails);
    std::printf("ihc warm: %d cases, %d failure(s)\n", cases, fails);
    return fails != 0;
}
