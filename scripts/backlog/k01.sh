#!/bin/bash
# k01.sh: K-01's lane sequence -- the fused dual launch (RADIANCE_HY4_FUSE_DUAL=1) and the two
# launches it replaces (=0), interleaved on the lane, stamped, with the greedy phase for the text;
# then the greedy logprobs probe per arm. Everything lands in runs/<lane>/k01-dual-*/.
#
#   scripts/backlog/k01.sh                                    the local lane, run.sh taking the pair
#   scripts/backlog/lane.sh iks-A 'scripts/backlog/k01.sh'     shipped and run in an iks lane
#
# K01_STAMP picks the RADIANCE_AQL_STAMP record (default 7).
set -o pipefail
here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
. "$here/lanes.sh"
RAD_REPO=$(rad_here); export RAD_REPO
lane_env "${RAD_LANE:-local}" || exit 2
stamp=${K01_STAMP:-7}
for tag in off1 on1 off2 on2; do
  case $tag in off*) v=0;; *) v=1;; esac
  echo "=== run $tag DUAL=$v $(date -u +%H:%M:%SZ)"
  "$RAD_REPO/scripts/backlog/run.sh" "k01-dual-$tag" nospec.yaml c1,c1b,c8,greedy \
    RADIANCE_AQL_STAMP=$stamp RADIANCE_HY4_FUSE_DUAL=$v 2>&1 | tail -20
  echo "--- run $tag done $(date -u +%H:%M:%SZ)"
done
for tag in off1 on1; do
  case $tag in off*) v=0;; *) v=1;; esac
  echo "=== logprobs $tag DUAL=$v $(date -u +%H:%M:%SZ)"
  "$RAD_REPO/scripts/backlog/lane.sh" --gpu "${RAD_LANE:-local}" \
    "bash '$RAD_REPO/scripts/backlog/k01_lp.sh' 'k01-dual-$tag' $v" 2>&1 | tail -12
done
echo "=== k01 lane sequence done $(date -u +%H:%M:%SZ)"
