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
- **S-04 (interleave EMA at the stager floor):** closed without code. On the served lane the decode-only run after a big
chunk does not lengthen (16 against 17.5 steps, medians over the two arrival windows) because since S-01 the samples
that hold the mean down are 1984-row chunks that cost 0.010 s, which a floor on rows keeps; where the change does act
it doubles the hold and costs the arrival's TTFT +0.29 s (1 decoder) and +0.92 s (4). Numbers, tables and commands:
`closed/S-04.md`.
