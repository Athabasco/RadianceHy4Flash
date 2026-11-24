#!/bin/bash
# k01_lp.sh <run-tag> <0|1>: the greedy logprobs probe against a server on the lane's rendered config
# with RADIANCE_HY4_FUSE_DUAL set, run through lane.sh --gpu so it holds the pair. The JSON lands
# beside the run in runs/<lane>/<run-tag>/lp-<0|1>.json (k01.sh's second arm).
set -o pipefail
run=${1:?usage: k01_lp.sh <run-tag> <0|1>}
val=${2:?usage: k01_lp.sh <run-tag> <0|1>}
here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
. "$here/lanes.sh"
RAD_REPO=$(rad_here); export RAD_REPO
lane_env "${RAD_LANE:-local}" || exit 2
D=$RAD_LANE_DIR/$run
out=$D/lp-$val.json
[ -f "$D/nospec.yaml" ] || { echo "k01_lp: no rendered cfg in $D" >&2; exit 2; }
RADIANCE_HY4_FUSE_DUAL=$val "$RAD_BIN" --config "$D/nospec.yaml" > "$D/lp-$val.server.log" 2>&1 &
P=$!
for i in $(seq 1 900); do
  lane_health "$RAD_PORT" && break
  kill -0 $P 2>/dev/null || break
  sleep 1
done
if kill -0 $P 2>/dev/null; then
  PROBE_DIR=$here python3 "$here/k01_lp.py" "$RAD_PORT" "$out"
else
  echo "k01_lp: the server did not answer /health" >&2
  tail -5 "$D/lp-$val.server.log" >&2
fi
kill $P 2>/dev/null
for i in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -9 $P 2>/dev/null; wait $P; echo "server-exit=$?"
