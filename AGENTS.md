# AGENTS.md — working on the Hy4-flash backlog from outside (no lane)

You are working on this repository without a GPU lane. Everything you need is in the tree;
read these before working:

- `notes/backlog/CARDS.md` — the working rules. §2 is the baseline and merge bar, §6 is the
  outside path (no lane). It is short; read all of it once.
- `notes/backlog/optimisation-backlog.md` — the items. Each item has an issue here; the issue
  is the unit of work, the PR is only its delivery.

## How an item is taken

1. Take an item through its issue: assign yourself (or comment `/take` with the tooling you
   have).
2. Open a PR whose title or body contains a closing keyword for the item's issue
   (`Fixes #N` / `Closes #N`). Without the keyword, the follow does not bind your PR to the
   issue and nobody reviews it (`#172` was missed exactly that way).
3. Title the PR `[needs lane] <item>: <subject>`: the code was checked without GPU; a lane
   measures it later and merges by §2.

## What a change carries

- The change **and its tests** (CARDS.md §6): a code path without a test in the same tree is
  a blocking finding (this came up on the router-bf16 PR: the branch lived in three sibling
  kernel files, only one had a test).
- Nothing in `third_party/` is edited; the engine, when involved, is touched only as a patch
  (repository rule).
- Commits, PR text and issue comments in English.

## What you can verify (and must, before pushing)

No GPU: the CPU build (`./build.sh` or CMake presets), `tests/r4d_emu/`, the libref / libavx
reference paths, `tests/chain_scalar_audit.py`, the plugin lints (`hyv4flash_arch_test`), the
scheduler tests (`sched_test`), every ctest that runs without a card. Run the touched ones;
paste nothing you did not run.

Do **not** claim measured GPU numbers — you have no lane. Instead, list inside the PR body
the item's "Verify" GPU checks and the `stamps` / `kbench` rows its merge bar wants; a lane
run here measures the PR (before / after on one lane) and merges or closes it by §2.

## How the review works here

- A Reviewer reads the PR at its head twice per finding round: once at push, again after your
  next push. Findings arrive as comments on the item's issue, with a verdict
  (`approve` / `changes`) and blocking vs judgement findings.
- Fix findings by pushing to the same branch. Never force-push: the reviews are head-pinned,
  and a rewritten history voids them.
- The owner merges after an approve. You never merge, never close the PR yourself.

## Feedback loop

If a verdict says `approve`, the merge closes the issue automatically. If it says `changes`,
the blocking findings are mandatory; judgement findings are advice unless the fix is trivial.
State in one line per finding what you changed about it, so the re-review can check it.
