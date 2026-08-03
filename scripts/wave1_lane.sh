#!/usr/bin/env bash
# Wave-1 dev-phase queue (prereg reports/wave1-anneal-decomposition-prereg.md §4),
# single-lane variant: GPU 0 (bus 03) only — GPU bus-01 is wedged ("GPU requires
# reset", 2026-07-23) and unreachable until a driver reset.
#
# Order: complete the seed-1511 block first (it is program #17's selection cell
# and program #18's sweep cell), then the 1512/1513 held-out blocks. Each run
# goes through the babysitter (retry + resume; forked tails re-fork on retry).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/wave1_lane.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() { echo "=== $(date -Is) START $*"; bash scripts/babysit_nanogpt.sh "$@"; echo "=== $(date -Is) END $*"; }

# ---- seed 1511: prefix, arms A/C, SF (kappa x rho) sweep, ramp -------------
run configs/dev/wave1_prefix.yaml      1511
run configs/dev/wave1_wsd_acc.yaml     1511
run configs/dev/wave1_constlr_acc.yaml 1511
run configs/dev/wave1_sf_k10_r07.yaml  1511
run configs/dev/wave1_sf_k10_r09.yaml  1511
run configs/dev/wave1_sf_k05_r07.yaml  1511
run configs/dev/wave1_sf_k05_r09.yaml  1511
run configs/dev/wave1_ramp.yaml        1511

# ---- seed 1512: prefix, arms A/C, ramp -------------------------------------
run configs/dev/wave1_prefix.yaml      1512
run configs/dev/wave1_wsd_acc.yaml     1512
run configs/dev/wave1_constlr_acc.yaml 1512
run configs/dev/wave1_ramp.yaml        1512

# ---- seed 1513: prefix, arms A/C --------------------------------------------
run configs/dev/wave1_prefix.yaml      1513
run configs/dev/wave1_wsd_acc.yaml     1513
run configs/dev/wave1_constlr_acc.yaml 1513

# SF confirm on 1512 uses the sweep-winning (kappa, rho) — launched separately
# after the seed-1511 sweep is read out.
echo "=== $(date -Is) WAVE1 LANE COMPLETE"
