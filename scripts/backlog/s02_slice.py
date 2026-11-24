#!/usr/bin/env python3
"""s02_slice.py <unsliced run dir> <sliced run dir>: design §8.2 for S-02b, on a card.

Two runs of the same served checkpoint, one with the hook off and one cutting each prefill-only
step into m sub-steps, on the same lane, the same probes, and the buffer probe on:

    RAD_LOG=trace RAD_CHUNK_SLICES_PREFILL=0 RADIANCE_DUMP_BUF=final_hidden,streams,block_in,hc_post \\
        scripts/backlog/lane.sh local 'scripts/backlog/run.sh s02-whole nospec.yaml greedy'
    RAD_LOG=trace RAD_CHUNK_SLICES_PREFILL=4 RADIANCE_DUMP_BUF=final_hidden,streams,block_in,hc_post \\
        scripts/backlog/lane.sh local 'scripts/backlog/run.sh s02-sliced nospec.yaml greedy'
    python3 scripts/backlog/s02_slice.py runs/local/s02-whole runs/local/s02-sliced

`run.sh` leaves the probe JSON and the server log in the run's directory. RAD_LOG=trace is what makes
the buffer probe (core/engine_probe.cpp, RADIANCE_DUMP_BUF) print, one `step N <buffer> ... n= hash`
line per read. What this compares, per dumped buffer:

  * every read of the run's *last* prompt's decode steps, in order, one for one: the last
    `gen - 1` reads, the prompt's own count of tokens it decoded (a load phase's `decode_steps`
    for a phase that ran one prompt). The probe prints no phase for a trunk step and a decode step
    is one read, so the prompt's own count is what separates them from the prefill steps before;
  * the chunk's end: the read immediately before those, i.e. the last prefill read -- the step in
    which the whole trunk has run and the chunk's rows are written. The sliced arm has m reads
    where the unsliced arm has one, so only the last of them is compared: the earlier ones are mid
    trunk, what the next sub-step resumes, and have no counterpart;
  * the greedy answers, when the phase recorded them: per prompt the text's sha256, its length and
    the generated count.

The chunk's KV itself has no dump: a decode row attends over the whole sequence's KV, so a chunk row
stored differently moves every read after it, which is why the decode reads are compared. The dumped
buffers (`streams`, `block_in`, `hc_post`, `final_hidden` -- what hyv4flash declares persistent) are
the state a slice hands on, so they carry the same weight.

A phase that ran more than one prompt (`greedy`) keeps its earlier prompts' decode reads before the
later prompts' prefill steps, so the comparison above is of its last prompt only (for `greedy`, the
5.7K arrival); the text comparison is what covers every prompt. Run a single-prompt phase (`c1`,
`c8`) per prompt when the dumps of each matter.

Exit status 1 when any comparison differs, 0 when all agree, so a lane can gate on it.
"""
import glob
import json
import os
import re
import sys

# One line of the buffer probe, as the log carries it (a Trace line: the level, the model/rank tag
# and the source site come first, and a colour escape can sit between them):
#   "T engine_probe.cpp:86  step 12        final_hidden  bf16  n=2048  hash 3f...  mean ..."
# A drafter's dump carries a phase word before the name; a trunk step's does not.
PROBE = re.compile(r"step (\d+)\s+(.+?)\s+(\S+)\s+n=(\d+)\s+hash ([0-9a-fA-F]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def probe_lines(path):
    """{buffer name: [(step, n, hash), ...]} in the log's order, and whether the log has any."""
    out, any_line = {}, False
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                m = PROBE.search(ANSI.sub("", ln))
                if not m:
                    continue
                any_line = True
                step, head, _dt, n, h = m.groups()
                # A buffer name is scoped ("tp0.final_hidden"): the leaf is what the run names.
                name = head.split()[-1]
                out.setdefault(name.rsplit(".", 1)[-1], []).append((int(step), int(n), h.lower()))
    except FileNotFoundError:
        return None, False
    return out, any_line


def run_files(d):
    """The run's probe JSON (run.sh writes <tag>.json) and its server log."""
    js = sorted(glob.glob(os.path.join(d, "*.json")))
    return (js[0] if len(js) == 1 else None), os.path.join(d, "server.log")


def phases(path):
    """{phase name: phase} from a probe.py JSON ({} when there is none)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, errors="replace") as f:
        doc = json.load(f)
    return {ph.get("name", "?"): ph for ph in doc.get("phases", [])}


def greedy(phs):
    """{prompt name: (sha256, chars, gen)} of the greedy phase, or None."""
    ph = phs.get("greedy")
    if not ph:
        return None
    out = {}
    for i, t in enumerate(ph.get("texts", [])):
        out[t.get("name", "#%d" % i)] = (t.get("sha256"), t.get("chars"), t.get("gen"))
    return out


def tail_decode(phs):
    """How many reads the run's *last* prompt decoded: one a token after the first, which its
    prefill step produced. From the last prompt's `gen` when the phase records the prompts (a
    greedy phase), else the phase's `decode_steps` (a load phase, one prompt). 0 when the JSON says
    neither."""
    for ph in reversed(list(phs.values())):
        texts = ph.get("texts") or []
        try:
            if texts:
                return max(0, int(texts[-1].get("gen") or 0) - 1)
            return max(0, int(round(float(ph.get("decode_steps") or 0))))
        except (TypeError, ValueError):
            pass
    return 0


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: s02_slice.py <unsliced run dir> <sliced run dir>")
    whole, sliced = sys.argv[1], sys.argv[2]
    bad = 0

    wj, ws = run_files(whole)[0], run_files(sliced)[0]
    wp, sp = phases(wj), phases(ws)
    if not wp or not sp:
        print("probe   MISSING  %s / %s (run.sh writes <tag>.json)" % (wj or whole, ws or sliced))
        return 1
    if sorted(wp) != sorted(sp):
        print("probe   DIFF  phases %s and %s: the two arms must run the same phases"
              % (",".join(sorted(wp)) or "-", ",".join(sorted(sp)) or "-"))
        bad = 1

    gw, gs = greedy(wp), greedy(sp)
    if gw is not None and gs is not None:
        names = sorted(set(gw) | set(gs))
        same = [n for n in names if gw.get(n) == gs.get(n)]
        if len(same) == len(names) and names:
            print("greedy  %d prompt(s) identical (text sha256, length, generated count)"
                  % len(names))
        else:
            bad = 1
            print("greedy  DIFF  %s" % ", ".join(sorted(set(names) - set(same))))
            for n in names:
                if gw.get(n) != gs.get(n):
                    print("  %-16s unsliced %s" % (n, gw.get(n)))
                    print("  %-16s sliced   %s" % ("", gs.get(n)))

    k_w, k_s = tail_decode(wp), tail_decode(sp)
    pw, pany = probe_lines(os.path.join(whole, "server.log"))
    ps, pany_s = probe_lines(os.path.join(sliced, "server.log"))
    if not (pw and ps):
        why = "no buffer-probe lines" if not (pany or pany_s) else "a run has none"
        print("dumps   MISSING  %s in one of the logs (RADIANCE_DUMP_BUF=..., and RAD_LOG=trace so "
              "the probe prints at Trace)" % why)
        return 1
    if k_w != k_s or k_w < 1:
        print("decode  DIFF  the runs' last prompts decoded %d and %d token(s)" % (k_w, k_s))
        bad = 1
    k = min(k_w, k_s)
    if k < 1:
        print("dumps   SKIP  the probe JSON does not say how many decode steps the runs made (it "
              "needs `decode_steps` or a prompt's `gen`)")
        return 1 if bad else 0
    for name in sorted(set(pw) | set(ps)):
        a, b = pw.get(name, []), ps.get(name, [])
        if not a or not b:
            bad = 1
            print("%-7s %-16s DIFF  %d vs %d read(s)" % ("dump", name, len(a), len(b)))
            continue
        a.sort()
        b.sort()
        da, db = a[-k:] if k else [], b[-k:] if k else []
        if len(da) != k or len(db) != k:
            bad = 1
            print("decode  %-16s DIFF  %d and %d read(s) before the last %d"
                  % (name, len(a), len(b), k))
        else:
            diff = next((i for i in range(k) if da[i][1:] != db[i][1:]), None)
            if diff is not None:
                bad = 1
                print("decode  %-16s DIFF  read %d (step %d vs %d): unsliced n=%d hash %s, sliced "
                      "n=%d hash %s" % (name, diff + 1, da[diff][0], db[diff][0], da[diff][1],
                                        da[diff][2], db[diff][1], db[diff][2]))
            else:
                print("decode  %-16s %d read(s) identical (steps %d..%d vs %d..%d)"
                      % (name, k, da[0][0], da[-1][0], db[0][0], db[-1][0]))
        # The chunk's end: the last read the prefill steps left, one step before the decode ones.
        if len(a) > k and len(b) > k:
            la, lb = a[-k - 1], b[-k - 1]
            if la[1:] != lb[1:]:
                bad = 1
                print("chunk   %-16s DIFF  step %d vs %d: n=%d hash %s vs n=%d hash %s"
                      % (name, la[0], lb[0], la[1], la[2], lb[1], lb[2]))
            else:
                print("chunk   %-16s n=%d hash %s identical (step %d vs %d; %d prefill read(s) "
                      "unsliced, %d sliced)" % (name, la[1], la[2], la[0], lb[0], len(a) - k,
                                                len(b) - k))
        else:
            print("chunk   %-16s no prefill read in %s (a prompt the cache served whole)"
                  % (name, "both runs" if len(a) <= k and len(b) <= k else
                     ("the unsliced run" if len(a) <= k else "the sliced run")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
