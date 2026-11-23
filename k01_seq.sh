#!/bin/bash
# K-01's lane sequence, run inside a lane (local or iks, shipped as the tree). Four served runs of
# RADIANCE_HY4_FUSE_DUAL=0 (the two launches) and =1 (the one dual launch), interleaved, stamped,
# with the greedy phase for the text; then the greedy logprobs probe per arm.
set -o pipefail
stamp=${K01_STAMP:-7}
for tag in off1 on1 off2 on2; do
  case $tag in off*) v=0;; *) v=1;; esac
  echo "=== run $tag DUAL=$v $(date -u +%H:%M:%SZ)"
  scripts/backlog/run.sh "k01-dual-$tag" nospec.yaml c1,c1b,c8,greedy \
    RADIANCE_AQL_STAMP=$stamp RADIANCE_HY4_FUSE_DUAL=$v 2>&1 | tail -20
  echo "--- run $tag done $(date -u +%H:%M:%SZ)"
done
for tag in off1 on1; do
  case $tag in off*) v=0;; *) v=1;; esac
  echo "=== logprobs $tag DUAL=$v $(date -u +%H:%M:%SZ)"
  scripts/backlog/lane.sh --gpu "${RAD_LANE:-local}" "bash $PWD/k01_lp.sh k01-dual-$tag $v $PWD/k01-lp-$v.json" 2>&1 | tail -12
done
echo "=== k01 lane sequence done $(date -u +%H:%M:%SZ)"
