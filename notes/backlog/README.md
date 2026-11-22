# Hy4-flash optimisation backlog

The Hy4-flash serving path's optimisation backlog, with the analysis it rests on. It covers TP2 on 2 x R9700 at 210 W and the
container `hy4-flash-iqr-v4-u3.rad` (keptn/hy4-flash-iqr-v4).

| file | what it is |
|---|---|
| `optimisation-backlog.md` | the backlog, copied unedited (status 2026-11-22, `src` @ c655f58c) |
| `audit/{A..E}-*.md`, `audit/COMMON.md` | the five read-only code audits the items cite: op lists, kernel fact sheets, thread and wake maps, longer derivations |
| `refs/kernels/*.md`, `refs/notes/*.md` | the earlier notes the backlog cites |
| `CARDS.md` | how cards built from the backlog work: lanes, merge bar, lossy items, working from outside |
| `../../scripts/backlog/` | the measuring tools and configs of §2, copied unedited (still with their local paths; the SETUP-LANES card makes them lane-aware) |

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
