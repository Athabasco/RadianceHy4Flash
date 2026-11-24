# S-02 (`--chunk-slices`, layer-sliced prefill chunk): CLOSED — the lever halves the bulk of the stall but keeps a 0.43-0.49 s tail, pays the arrival 26 % of its TTFT, and doubles the expert copy

Item as written (backlog S-02): plan a chunk as m layer-sliced sub-steps so the decoders ride a slice of the chunk's
layers; gain "stall 0.5-0.8 s -> ~0.15-0.2 s at m = 4 with prefill rate and TTFT unchanged"; verify: greedy continuation
byte-identical against unsliced, `sched_test` shapes, the stall probe, BetterBench. Design and its measurement plan:
`notes/backlog/design/S-02.md` (§5 the stager's cost, §10 the A/B this note runs, §12 S-02b's Verify).

## The number

One lane, arms back to back in one call: `iks-A` (TP2, `nospec.yaml`, host-pool 16000 MiB, port 8042), 2026-11-24,
`probe.py` phases `stall1,stall4,pf6,pf20,greedy`. `stall1` is one decoder and a 5.7K-token arrival, `stall4` four
decoders and the same arrival; a gap is the decoder's wait between two of its own steps (ms), and the arrival TTFT is
the one the lever is trying to protect.

| arm | `stall1` arrival TTFT | gaps, 1 decoder | gaps, 4 decoders | sliced steps / sub-steps | staged, `stall1` window |
|---|---|---|---|---|---|
| `chunk-slices: off` | 2.070 s | 34.7 39.1 40.2 40.7 40.7 **428.7 472.3 538.9** | 431.0-562.5 | 0 / 0 | 123 layers, 12.16 GiB |
| `auto` (planned m = 2) | 2.185 s | 36.3 40.6 41.1 59.4 **341.7 367.1 431.9 476.6** | 330.6-434.2, one 727.3 | 1 / 2 | 164 layers, 16.20 GiB |
| `chunk-slices: 4` | 2.608 s | 41.2 44.1 **241.5 249.8 265.6 271.4 429.8 473.7** | 230.1-282.6, tail 433.9-493.9 | 1 / 4 | 246 layers, 24.31 GiB |

The lever does what it was built to do, and the number is in the middle column: at m = 4 the decoder's bulk wait is a
quarter of the chunk, 0.43-0.56 s -> 0.23-0.28 s. It is not the item's number, for three reasons, and the third is the
one that matters:

1. **`auto` does not plan m = 4.** It plans 2 on both lanes (sub-steps 2 for one chunk). The m law targets 150 ms
   against a ~1 s chunk step, which asks for 6-7; `--chunk-slice-cost-pct 10` holds it at 2, so the default setting of
   the lever lands at 0.34-0.43 s, not 0.15-0.2 s. §10 named this: "If the rate falls and the stall does not, the m
   law's cost cap is wrong".
2. **What is left of the stall is not the chunk.** A ~0.43-0.49 s gap survives in every arm — 428.7 / 472.3 unsliced and
   429.8 / 473.7 at m = 4, the same two gaps to within 5 ms. Slicing the chunk cannot shorten a cost that is not the
   chunk's compute; it sits at the arrival's boundary and is its own item.
3. **Slicing doubles the expert copy** (§10's first check, and §5's premise). Over the same arrival window the unsliced
   arm stages 123 layers / 12.16 GiB and m = 4 stages 246 / 24.31 GiB (m = 2: 164 / 16.20) — two full trunk passes per
   arrival instead of one. The chunk's slices stage their own layers and the decode rows keep traversing the whole
   trunk between slices, so each evicts the other from the ring: "the group's first slice pays the unsliced chunk's
   copy and the later slices pay little" does not hold at 16000 MiB of host pool (the local lane at 39000 reads the
   same ratio, 123 / 12.68 GiB against 246 / 25.11). §10's answer to that failure is §5's fallbacks — stage per slice
   and keep the decode rows' experts resident across the slice boundaries — and "no other number on this page applies"
   until they are built.

The e2e side, same lane: `pf6` 1.838 / 1.851 / 1.845 s and `pf20` 6.834 / 6.893 / 6.865 s (off / auto / 4). A
prefill-only step is never sliced (`only_a_mixed_step_with_one_chunk_is_sliced`), so those rows say the lever costs
nothing when there is no decoder to protect, and they are why the item's "prefill rate unchanged" holds. The arrival
TTFT does not: 2.070 -> 2.608 s at m = 4 (+26 %, and the extra copy above is where it goes) and 2.185 s at `auto`.

Byte identity holds everywhere: the four `greedy` continuations (256 / 256 / 102 / 64 tokens) are the same four
sha256 in all three arms here, and the same four again in the three local arms —
`d6da4fb3…`, `0699709c…`, `e7a23c5d…`, `466e53a2…`.

The local lane (`local`, TP2, host-pool 39000, port 8041, same phases, back to back) is reported because it was run, not
used for the verdict: the box was at load average 85 with other cards' builds, so `stall1` reads 565.4 / 494.8 / 469.6 ms
worst for off / auto / 4 — the same shape as iks — while its unsliced `stall4` arm queued the arrival for 24.5 s and
showed 3.46-4.54 s decoder gaps, which is the host, not the lever.

## Why it does not carry

The item's gain is one sentence — "0.5-0.8 s -> ~0.15-0.2 s at m = 4 with prefill rate and TTFT unchanged" — and two of
its three clauses fail: the stall lands at 0.23-0.28 s in the bulk with a 0.47 s tail, and the arrival's TTFT rises a
quarter. What the lever does deliver (byte identity, a quarter-length bulk wait, no prefill cost) is real, but it is
paid for with a copy that doubles, and the copy is the part §5 said would be free.

## What stays and what comes next

- The S-02b code is in the branch already (this card's earlier commits) and the lever is **off by default**, so nothing
  a serve does today changes: `--chunk-slices` stays a lever to be turned on deliberately. Whether to keep it, and
  what to build on it, is the owner's call at merge; this note gives the numbers either way.
- The two review findings of this round are independent of that verdict and stay: the scheduler's prefill-only
  refusal now has its test, and a calibration window (`--calib-dir`/`--calib-layers`) keeps a chunk whole at declare
  and at plan instead of failing to bring the model up.
- The next work is §5's fallbacks, in this order: (a) the stager keeps the decode rows' experts resident across the
  slice boundaries, so the trunk is traversed once per group rather than once per side; (b) the m law's cost cap,
  which plans 2 where the target asks for 6-7; (c) the ~0.43-0.49 s boundary gap, which is not the chunk.

The harness this card measured with stays: `scripts/backlog/probe.py`'s `stage` sampler records `stage_layers`,
`staged_bytes` and `stage_copies` from `/stats` across an arrival, and `steps_sliced` / `substeps` from `/metrics` per
phase, so the copy share per slice that this note reports is one run away on any lane.

Commands, both lanes (`local` is `scripts/backlog/lane.sh local`, `iks-A` tars the tree and runs the same in
`~/moebius-gpu/RadianceHy4Flash-run-1954/src`):

    RAD_CHUNK_SLICES=off  scripts/backlog/run.sh s02off  nospec.yaml stall1,stall4,pf6,pf20,greedy
    RAD_CHUNK_SLICES=auto scripts/backlog/run.sh s02auto nospec.yaml stall1,stall4,pf6,pf20,greedy
    RAD_CHUNK_SLICES=4    scripts/backlog/run.sh s02m4   nospec.yaml stall1,stall4,pf6,pf20,greedy
