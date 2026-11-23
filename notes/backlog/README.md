# Hy4-flash optimisation backlog

The Hy4-flash serving path's optimisation backlog, with the analysis it rests on. It covers TP2 on 2 x R9700 at 210 W and the
container `hy4-flash-iqr-v4-u3.rad` (keptn/hy4-flash-iqr-v4).

| file | what it is |
|---|---|
| `optimisation-backlog.md` | the backlog, copied unedited (status 2026-11-22, `src` @ c655f58c) |
| `audit/{A..E}-*.md`, `audit/COMMON.md` | the five read-only code audits the items cite: op lists, kernel fact sheets, thread and wake maps, longer derivations |
| `refs/kernels/*.md`, `refs/notes/*.md` | the earlier notes the backlog cites |
| `CARDS.md` | how cards built from the backlog work: lanes, merge bar, lossy items, working from outside |
| `closed/<ID>.md` | items closed without code: the number, the reason, and what an operator or a later card needs from them |
| `../../scripts/backlog/` | the measuring tools and configs of §2: `run.sh`, `probe.py`, `bench.py`, `agg.py`, `layer.py`, `profdiff.py`, `profagg.py`, `attr.py`, `prompts.py`, `demangle.py` and the configs, with `lane.sh`, `lanes.sh`, `lane-cfg.py` and `lane-pair.sh` that make them lane-aware (§1.1) |

How paths in the backlog map onto this directory:

- `src/<path>`: the repository root, `<path>`.
- `work/backlog/audit/X.md`: `notes/backlog/audit/X.md`.
- `kernels/X.md`: `notes/backlog/refs/kernels/X.md`.
- `notes/X.md`: `notes/backlog/refs/notes/X.md`, except `notes/sweep.md`, which is the repository's own.
- `work/backlog/{run.sh,*.py,*.yaml}` and `work/u3/bench.py`: `scripts/backlog/`.
- Other `work/...` files (raw stamp CSVs, server logs) stayed on the measuring host and are not needed to build an item.

Line numbers in the backlog were read at c655f58c. Find them again at HEAD before you edit.

Since the backlog was written, these decisions were made:

- **Format changes:** none. F-01..F-11, F-13 (as a format change), F-18, F-20 and F-21 are dropped.
- **"The friend's decisions":** §4 calls the lossy rows the friend's. They are the repository owner's decisions now.
- **Lossy runtime rows:** X-07 (with F-12 / F-13 as its groups), F-14, F-15, F-16, F-17 and P-13 / F-19 are built opt-in, each with a KL report.
- **M-13 (CWSR off on a dedicated box):** closed without code. The operator setting is worth ~68 MiB = 48 units, but
the engine refuses to serve under it (Incompatible) and its failure mode is a full card reset, so it buys the smallest
of the three lossless memory levers at the highest price. Number, sources and the verify steps a lane would run:
`closed/M-13.md`.
- **S-01(b) (`--decode-headroom`, FP-3):** closed without code. S-01(a)'s `checkpoint-interval: 1984` already leaves no
remainder step in the served schedule (three `MIXED` steps an arrival, both arms of every run), so at the served geometry
the flag has no step to remove and only grows the arena (1.39 -> 1.43 GiB a rank, +43.34 MiB); at 2048 rows it reaches
exactly the schedule (a) reaches, inside noise and with a larger arena. The cherry-pick is reverted, the harness it
needed stays. Numbers, lanes and what a later card would need: `closed/S-01b.md`.
- **M-03a (stager arms at 1024 rows, the busy-chunk floor's off-by-one):** closed without code. In the served config a
step of 1024 rows has no ring to copy into, so `min_rows` is not the lever, and where the floor binds (two 1024-row
chunks a step) the step is 2048 rows and already staged: the lane measured the same `stage_layers`, the same h2d bytes
and the same ms before and after rounding the floor up. Number, lane and runs: `closed/M-03a.md`.
