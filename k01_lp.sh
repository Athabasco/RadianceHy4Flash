#!/bin/bash
# k01_lp.sh <run-tag> <0|1> <out.json>: the greedy logprobs probe against a server on the lane's
# rendered config with RADIANCE_HY4_FUSE_DUAL set, run through lane.sh --gpu so it holds the pair.
set -o pipefail
run=$1; val=$2; out=$3
S=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
. "$RAD_REPO/scripts/backlog/lanes.sh"
D=$RAD_LANE_DIR/$run
[ -f "$D/nospec.yaml" ] || { echo "k01_lp: no rendered cfg in $D" >&2; exit 2; }
RADIANCE_HY4_FUSE_DUAL=$val "$RAD_BIN" --config "$D/nospec.yaml" > "$D/lp-$val.server.log" 2>&1 &
P=$!
for i in $(seq 1 900); do
  lane_health "$RAD_PORT" && break
  kill -0 $P 2>/dev/null || break
  sleep 1
done
if kill -0 $P 2>/dev/null; then
  PROBE_DIR=$RAD_REPO/scripts/backlog python3 "$S/k01_lp.py" "$RAD_PORT" "$out"
else
  echo "k01_lp: the server did not answer /health" >&2
  tail -5 "$D/lp-$val.server.log" >&2
fi
kill $P 2>/dev/null
for i in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -9 $P 2>/dev/null; wait $P; echo "server-exit=$?"
