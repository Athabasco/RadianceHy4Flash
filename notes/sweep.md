# Hy4-flash scheduler sweep vs. arrival stalls (2026-11-22)

Snapshot binary, u3 container, TP2, 210 W, base work/u3/coupled03.yaml; BetterBench --quick --complete --workload --prefill
--workload-streams 2 4 --workload-duration 90. Results: bbench/res/sweep-<tag>.{json,html}, logs work/sweep/bb-<tag>.log.
Table and recommendation: docs/ARCHITECTURES.md Hy4 "TP2 recipe" (stalls while a prompt arrives).
Tags: auto b1024 b512 b256 i0 i2 ia2 b1024ia2 m300 m150, repeats b512r m300r (repeat within 10 ms of stall, 1 s of TTFT).
Single runs per depth are 1-2 arrivals; treat TTFT per depth as coarse. Cold prefill identical in all runs.
Best: b512 (balanced), m300 (no warning), HTML: /root/work/claude-scratch/bbench/res/sweep-b512.html
