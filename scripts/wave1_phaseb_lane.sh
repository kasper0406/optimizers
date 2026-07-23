#!/usr/bin/env bash
# Wave-1 Phase B (program #18) — gate record reports/wave1-phase-b-gate.md.
# Per eval seed: prefix P -> arm C (constant-LR + accumulators) -> arm B (SF
# winner k=1.0, rho=0.7). Arm A = existing n=10 baseline (no new runs).
# Eval seeds are passed via --seed per CLAUDE.md ground rule 2.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { echo "=== $(date -Is) START $*"; bash scripts/babysit_nanogpt.sh "$@"; echo "=== $(date -Is) END $*"; }

for seed in 1710 1711 1712 1713; do
  run configs/dev/wave1_prefix.yaml      "$seed"
  run configs/dev/wave1_constlr_acc.yaml "$seed"
  run configs/dev/wave1_sf_k10_r07.yaml  "$seed"
done
echo "=== $(date -Is) PHASEB LANE COMPLETE"
