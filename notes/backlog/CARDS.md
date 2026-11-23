# Cards from the Hy4-flash backlog: rules

Each card is one item of `optimisation-backlog.md` (or a tight pair of items). Its issue quotes the item in full, so the issue can
be read without this directory. The audits under `audit/` hold the longer derivations.

## 1. Lanes

A lane is a TP2 pair of R9700 at 210 W with the u3 container (`hy4-flash-iqr-v4-u3.rad`, keptn/hy4-flash-iqr-v4).
The lane is the only thing a run needs to know: `scripts/backlog/lane.sh` gives a command the lane's port, model,
tiers and cards, so the same tools and the same `probe.py` phases run on every lane.

| lane | where | cards | port | Engram, host pool |
|---|---|---|---|---|
| `local` | the Moebius host, its two cards | `<lane lock dir>/gpu.lock`, then `gpu2` | 8041 | `ngram-placement: ram` (a pinned copy), host pool 39000 MiB: the served config, `scripts/backlog/*.yaml` as they are |
| `iks-A` | `ssh iks`, a free pair of its four | `gpu2` on iks | 8042 | `ram=/dev/shm/radiance-ngram` (one copy in tmpfs, pinned by this lane), host pool 16000 MiB |
| `iks-B` | `ssh iks`, the other pair | `gpu2` on iks | 8043 | `disk=~/ngram` (a 60 GiB file on the NVMe, read through the row cache), host pool 16000 MiB |

Why the iks lanes are not the local config: iks has 112 GB of RAM and one u3 server pins ~60 GiB of Engram tables,
so two servers at `ram` do not fit, and the driver there has `no_system_mem_limit=N`, so one `/dev/shm` copy pinned
by both engines is refused too (each engine's pin counts against the cap). `iks-A` therefore keeps the single pinned
copy and `iks-B` reads the same table from a file; both take a host pool of 16000 MiB instead of 39000 so that the
two of them fit beside the 60 GiB. §1.1 has what that measured.

- **Server lane:** a full served A/B (stamps, tok/s, TTFT, KL).
- **Kernel lane:** selftest, rad-kbench, emu and the perf modes, on any free pair.
- **One lane per A/B:** a card's before and after run on the same lane and are compared against that lane's baseline,
  `baseline/<lane>.md`. Record the lane in the commit message or note.
- **Lanes differ:** iks-B reads Engram from disk and both iks lanes have a smaller host pool, so do not compare
  numbers across lanes -- not even iks-A with iks-B, which are two cards of the same machine but not two of the same
  table placement.
- **Power cap:** read the 210 W cap back before every run: `lane.sh` does it in `run.sh` and `lane-pair.sh` and
  prints `cap` per card; a card that is not at 210 W means the run's numbers are not the lane's.

### 1.1 Using the lanes (`scripts/backlog/`)

```sh
cd <tree>
scripts/backlog/lane.sh lanes                     # local iks-A iks-B
scripts/backlog/lane.sh --env iks-A               # what a lane exports: port, model, tiers, paths (`~` on iks)
```

- **`lane.sh <lane> <command>`** runs a shell command in the lane (or a program with its arguments). `local` runs it
  here; `iks-A` and `iks-B` tar the tracked and untracked tree to `ssh iks moebius-gpu/run.sh <name> <b64>` and run
  it in `~/moebius-gpu/<name>/src`, where the `build/` directory survives between calls, so the first iks call of a
  tree builds and the later ones are incremental. The tree name is `<repo>-<run>` (`RadianceHy4Flash-run-1638`);
  `RAD_IKS_NAME` overrides it. One iks call runs at a time (the unpack replaces the tree); two lanes together go
  through `lane-pair.sh`.
- **`lane.sh --gpu <lane> <command>`** the same, on the lane's pair: `flock <lock dir>/gpu.lock gpu2` on `local`,
  `gpu2` on iks. Nothing else takes a card: `run.sh` and `lane-pair.sh` start their servers this way. `gpu.lock`
  (default `<lane lock dir>` = `/tmp/radiance-lanes`, `RAD_LANE_LOCK_DIR` overrides) is what a local lane run holds
  for its whole run.
- **`run.sh <tag> <cfg.yaml> <phases> [ENV=VAL ...] [-- radiance args]`** one server in the lane, `probe.py`
  against it, then it is stopped. Everything lands in `runs/<lane>/<tag>/`: the lane's rendered config
  (`<cfg>.yaml` with the lane's port, model, host pool and `ngram-placement`), `server.log`, `run.out`, `<tag>.json`.
  `RADIANCE_AQL_STAMP=1` also leaves `<tag>.stamps.csv` there, the input of `layer.py` and `agg.py`. A lane whose
  `ngram-placement` is `ram=DIR` first clears that directory: the pinned copy belongs to one process and the one a
  killed run leaves behind refuses the next start (§1.2).
  ```sh
  scripts/backlog/lane.sh local 'scripts/backlog/run.sh c1 nospec.yaml c1,c8'
  scripts/backlog/lane.sh iks-A 'scripts/backlog/run.sh c1 nospec.yaml c1,c8'
  ```
- **`lane-pair.sh <cfg.yaml> <phases> [ENV=VAL ...]`** both iks lanes at once with the same config, both servers up
  together, `probe.py` against each, both stopped: the two pairs of one machine side by side. Run it inside an iks
  tree: `scripts/backlog/lane.sh iks-A 'scripts/backlog/lane-pair.sh nospec.yaml c1,c8'`.
- **`probe.py <port> <out.json> <phases>`** the phases a card quotes:
  `c1`, `c8` (one and eight decoders), the same on a DFly2 config (`coupled03.yaml`, `cpv.yaml`, `cprof.yaml`) for
  the drafter's `c1` / `c8`, `pf6` (~6.7K-token prefill: `22,000` characters of tree source) and `pf20` (~24K:
  `72,000`), `stall1` / `stall4` (a 5.7K arrival on one and on four running decoders), and the small prompts `pf2`
  (`7,400`), `tailN` (`N` characters, e.g. `tail400`) and `c1s` / `c8s`. It prints each phase as JSON and writes
  `out.json`; phases are comma-separated (`c1,c8,pf6,pf20,stall1,stall4,pf2,tail400`), `nowarm` skips the two warmup
  requests, and an unknown phase is refused instead of leaving a hole in the run. Its requests are sampled at
  `RAD_TEMP` (default 0.9, seeded per request); a host whose `TEMP` is a directory is not a temperature and is
  ignored.
- **`bench.py <port> <tag> <out-dir>`** the served benchmark (decode, prefill, cold/warm, `--reps`, `--temp`), and
  **`prompts.py`** holds the prompts and the code corpus both tools send: they are read from the tree the tools sit
  in (`RAD_SRC` overrides), never from a copy on the measuring host, so a lane's prompt is the same text everywhere.
- **`layer.py <stamps.csv> <probe.json> <phase> <rec> [<li>]`**, **`agg.py <stamps.csv> <probe.json> [--top N]`**,
  **`attr.py <stamps.csv> <probe.json> [--top N]`**: the per-layer and per-op tables from a run's stamps. They are
  plain CPU tools over those two files, so they run wherever the files are: on the lane, or here over a file copied
  back from iks with `scp iks:moebius-gpu/<name>/src/runs/<lane>/<tag>/<tag>.stamps.csv .`. The iks trees' `runs/`
  are not kept between calls (the tree is unpacked fresh each call), so copy back what a card needs.
- **`profdiff.py <log> [rank] [min-ms] [top]`** and **`profagg.py <log> [rank]`**: two runs of a `profile-ops: true`
  config (`prof.yaml`, `cprof.yaml`) compared op by op.
- The configs are the served (local) ones: `nospec.yaml` (no drafter), `coupled03.yaml` (DFly2 coupled, 5 proposals),
  `cpv.yaml` (the same with debug logging), `cprof.yaml` (DFly2 + `profile-ops`), `nsv.yaml` / `nsv1984.yaml`
  (debug logging, and a 1984-token checkpoint interval), `prof.yaml` (`profile-ops`). `lane.sh` rewrites only a
  lane's four keys, so a config's other lines mean the same thing on every lane.

### 1.2 What the lanes measured

Every lane was run on 2026-11-23 with the u3 container and the 210 W cap read back first
(`lane_begin`: 210 W on all 8 cards of iks and on both cards here).

**Both iks lanes served at once.** `scripts/backlog/lane.sh iks-A 'scripts/backlog/lane-pair.sh nospec.yaml c1,c8,pf6'`:
`iks-A` answered `/health` on 8042 after 85 s (it fills 60 GiB of tmpfs first), `iks-B` on 8043 after 1 s, both
stayed up and each decoded while the other was idle:

| run | `iks-A` Engram `ram=/dev/shm/radiance-ngram` | `iks-B` Engram `disk=~/ngram` |
|---|---|---|
| `c1` — 1 decoder, 256 tokens | 72.8 tok/s, 13.5 ms/step | 59.8 tok/s, 16.1 ms/step |
| `c8` — 8 decoders, 256 tokens each | 313.0 tok/s agg | 184.8 tok/s agg |
| `pf6` — 7,077-token prompt | ttft 1.99 s | ttft 9.67 s |

Memory with both up is the whole story: 112 GB total, `MemAvailable` 10.7 GiB before the probes and 10.4 GiB
after (the 60 GiB tmpfs table and 70 GB of page cache are not reclaimable on that host, and there is no swap),
so the two servers fit with about 10 GiB to spare and nothing else may run on iks beside them. `iks-B` is
slower because every Engram lookup reads a file (page cache, then NVMe) instead of a pinned copy: -18% at
`c1`, -41% at `c8`, 4.9x the ttft of a 7,077-token prompt. That is the placement, not the cards, which is why
§1 says not to compare numbers across lanes.

**The phases on each lane**, all rc=0, `server-exit=0`:

| phase | `local` (`run.sh nospec nospec.yaml …`) | `iks-A` alone (`run.sh fullA …`, the same phase list) |
|---|---|---|
| `c1` | 71.6 tok/s | 72.7 tok/s (13.5 ms/step) |
| `c8` | 287.1 tok/s agg | 311.3 tok/s agg |
| `pf2` — 1,896 tokens | ttft 0.60 s | ttft 0.42 s |
| `pf6` — 7,077 tokens | ttft 2.88 s | ttft 1.98 s (7,076 tokens) |
| `pf20` — 20,731 tokens | ttft 6.65 s | ttft 5.90 s |
| `stall1` — 7,905 tokens behind 1 decoder | arrival ttft 4.43 s | 4.21 s |
| `stall4` — 7,203 tokens behind 4 decoders | arrival ttft 3.92 s | 3.80 s |
| `tail400` — 193 tokens | ttft 0.15 s | ttft 0.13 s |

The `local` lane with the drafter, `run.sh fly2 coupled03.yaml c1,c8 RADIANCE_AQL_STAMP=1`:
`c1` 113.1 tok/s (2.81 tok/step), `c8` 333.3 tok/s agg (20.28 tok/step). The same run left
`runs/local/fly2/fly2.stamps.csv` (42 MB), and `agg.py` and `layer.py` read it there:
`agg.py runs/local/fly2/fly2.stamps.csv runs/local/fly2/fly2.json --top 5` splits the pass kinds
(303 records: 92 passes, 4.67 ms median wall, `mvm_kernel` 34.8%, `gemm_i8_kpar` 24.3%; 1,335 records:
91 passes, 17.22 ms, MoE routed gate_up 22.0%), and `layer.py … c1 1335 6` lists the per-op times of layer 6.
An iks lane's stamps are the same file: `lane.sh iks-A 'run.sh st nospec.yaml c1 RADIANCE_AQL_STAMP=1'` left 73 MB of
`st.stamps.csv` in the iks tree, `scp` brought it back, and `agg.py` read `c1`'s one pass kind (1,330 records: 255
passes, 13.49 ms median, trunk linears 18.4%, int8/w4 GEMMs 15.3%, all-reduce 12.1%).

What the lanes pinned down beyond the numbers:

- The u3 container on iks (`~/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad`, 138,093,449,672 bytes) hashes to
  `c4a80918139f2bae4d097976fd2794ca3f1f56e9ea7eb60600188c45f4f5e7aa`, the value in its `.sha256` from Hugging
  Face's LFS oid, and the same size as `/srv`'s copy. The download's `dl-u3.log` ends in `DONE`.
- A `ram=DIR` Engram table is one process's pinned copy and is not reused: a copy left in `/dev/shm` by an
  earlier run is 60 GiB of tmpfs that the next start counts against `MemAvailable` and refuses to start beside
  ("50.75 GiB of memory is available … this wants 45.49 GiB"). `lane_ngram_reset` clears it before a lane
  starts, so `iks-A` refills 60 GiB on each start (part of the 85 s) while `iks-B`'s `disk=` table is reused.
- `lane-pair.sh` is the only way two lanes run on iks at once: `lane.sh` sends the tree to one
  `~/moebius-gpu/<name>/src` and an unpack replaces it, so single-lane calls are serialised by
  `flock <lane lock dir>/iks.lock`.
- iks has no `curl`, so `lane_health` asks `/health` with `python3`; and a measuring host whose `TEMP` is a
  directory is not a sampling temperature (`RAD_TEMP` is).

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
