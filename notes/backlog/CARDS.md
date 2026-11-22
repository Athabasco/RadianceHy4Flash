# Cards from the Hy4-flash backlog: rules

Each card is one item of `optimisation-backlog.md` (or a tight pair of items). Its issue quotes the item in full, so the issue can
be read without this directory. The audits under `audit/` hold the longer derivations.

## 1. Lanes

A lane is a TP2 pair of R9700 at 210 W with the u3 container.

| lane | where | lock / scheduler | notes |
|---|---|---|---|
| `local` | the Moebius host, its two cards | `flock /root/work/claude-scratch/gpu.lock`, port 8041 | Engram in RAM (`ngram-placement: ram`): the served config |
| `iks-A`, `iks-B` | `ssh iks`, two of its four cards each | `~/moebius-gpu/run.sh` + `gpu2` (a lock per card) | reduced config: Engram from disk or a shared `/dev/shm` copy, smaller host pool, so two servers fit in 112 GB |

- **Server lane:** a full served A/B (stamps, tok/s, TTFT, KL).
- **Kernel lane:** selftest, rad-kbench, emu and the perf modes, on any free pair.
- **One lane per A/B:** a card's before and after run on the same lane and are compared against that lane's baseline,
  `baseline/<lane>.md`. Record the lane in the commit message or note.
- **Lanes differ:** iks reads Engram from disk, so do not compare numbers across lanes.
- **Power cap:** read the 210 W cap back before every run.

## 2. Merge bar (lossless items)

Merge when all three hold:

1. The output is byte-identical, or inside the item's stated bound (ORD / PF; see the field key at the top of the backlog's §3).
2. The kernel, op or step the item targets is measurably faster in ms (AQL stamps, rad-kbench or profile-ops) on its lane.
3. End-to-end tok/s or TTFT is not worse beyond noise.

A kernel gain whose end-to-end effect is inside noise still merges: gains compound.

Otherwise write `closed/<ID>.md` with the numbers and the reason, and close the card without merging code (like the backlog's §5).
Measurement-only items (K-17, C-01, T-03, M-14, M-15, ...) merge their note, plus the fix when the measurement calls for it.

## 3. Lossy items

- The change sits behind a switch, off by default.
- Report keptn's KL on u3 (mean / p99 KL, top-1, against today's default) plus tok/s / TTFT on one server lane in
  `lossy/<ID>.md`.
- Only the lossy review card proposes defaults, and the repository owner decides.

## 4. Out of scope

- Format or container changes: F-01..F-11, F-13 as a format change, F-18, F-20, F-21.
- The closed items of §5, P-15 (closed in refs/notes/p15.md) and D-08 (a pointer only).
- M-12 is folded into K-19, P-10, P-11 and P-12. D-10's parts are in K-08, K-16, K-18 and the D-10 card (`gemm_nt_f32`).

## 5. Parallel work

Cards run in parallel even when they touch the same file; whoever merges second rebases. `blocked_by` holds only real logical
dependencies.

`[design]` cards write `design/<ID>.md` and may rewrite the bodies of the build cards that wait for them. Run them with the
design agent: Opus 5.5, thinking medium.

## 6. Working from outside (no lane)

A worker without access to the lanes can still do a card:

1. Write the change and its tests.
2. Run what needs no GPU:
   - the build (`./build.sh` or the CMake presets)
   - the CPU emulation tests in `tests/r4d_emu/`
   - the libref / libavx reference paths
   - `tests/chain_scalar_audit.py`
   - the plugin lints (`hyv4flash_arch_test`)
   - the scheduler tests (`sched_test`)
   - every ctest that runs without a card
3. Push a branch and open a PR titled `[needs lane]`. List in it the GPU checks from the item's "Verify" part, and the stamps
   or kbench rows that the merge bar needs.
4. A lane run here measures the PR (before / after on one lane) and merges or closes it by §2.

Items whose whole value is a measurement (marked "lane required" in their issue) need a lane from the start.
