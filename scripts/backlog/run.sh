#!/bin/bash
# run.sh <tag> <cfg.yaml> <phases> [ENV=VAL ...] [-- extra radiance args]: one server (snapshot build) under the GPU lock, probe.py, stop.
cd /root/work/claude-scratch/hy4flash/work/backlog
B=$PWD/snap
tag=$1; cfg=$2; ph=$3; shift 3
export RADIANCE_HOME=$B/radiance_home
X=""
while [ $# -gt 0 ]; do if [ "$1" = "--" ]; then shift; X="$*"; break; fi; export "$1"; shift; done
flock /root/work/claude-scratch/gpu.lock bash -c "
  date
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do echo cap \$(cat \$f); done
  $B/bin/radiance --config $PWD/$cfg $X > $tag.server.log 2>&1 & P=\$!
  for i in \$(seq 1 900); do curl -sf http://127.0.0.1:8041/health >/dev/null && break; kill -0 \$P 2>/dev/null || break; sleep 1; done
  date
  kill -0 \$P 2>/dev/null && python3 probe.py 8041 $tag.json $ph 2>&1
  kill \$P 2>/dev/null; for i in \$(seq 1 90); do kill -0 \$P 2>/dev/null || break; sleep 1; done; kill -9 \$P 2>/dev/null; wait \$P; echo server-exit=\$?
  date
" > $tag.out 2>&1
