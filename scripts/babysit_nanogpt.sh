#!/usr/bin/env bash
# Failure + restart supervisor for local nanogpt runs (flaky-GPU tolerant).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/babysit_nanogpt.sh <config.yaml> <seed> [seed...]
#
# For each seed, in order:
#   - skip it if a completed results JSON for this config already exists
#     (val_curve reaches the final step);
#   - otherwise run scripts/run.py under a hang backstop (`timeout`); the
#     port's seed-keyed checkpoint (every 250 steps, resume: true) makes a
#     killed or crashed run resume where it left off;
#   - on failure, retry with escalating backoff up to MAX_ATTEMPTS; a card
#     that wedges hard therefore costs retries, never silent data loss.
#
# stdout carries only supervision events (one line each) so it can be
# streamed as a monitor; per-run training output goes to logs/babysit/.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CFG="$1"; shift
SEEDS=("$@")
GPU="${CUDA_VISIBLE_DEVICES:-?}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-9000}"   # seconds; full run ~4500s + compile margin
BACKOFFS=(30 60 120 300 600 600 600 600)
CFG_TAG="$(basename "$CFG" .yaml)"
LOGDIR=logs/babysit
mkdir -p "$LOGDIR"

is_done() {
  uv run python - "$1" "$CFG_TAG" <<'PY'
import glob, json, sys
seed, tag = int(sys.argv[1]), sys.argv[2]
for f in sorted(glob.glob(f"results/nanogpt_seed{seed}_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    cfg = d.get("config") or {}
    if tag not in str(cfg.get("path", "")):
        continue
    m = d.get("metrics") or {}
    curve = m.get("val_curve") or []
    ng = cfg.get("contents", {}).get("nanogpt", {}) or {}
    want = ng.get("num_iterations", 1750)
    if ng.get("max_steps") is not None:
        want = min(want, ng["max_steps"])
    if curve and curve[-1].get("step") == want and m.get("final_val_loss") is not None:
        sys.exit(0)
sys.exit(1)
PY
}

overall_rc=0
for seed in "${SEEDS[@]}"; do
  if is_done "$seed"; then
    echo "[gpu$GPU] seed $seed already complete — skipping"
    continue
  fi
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
      echo "[gpu$GPU] seed $seed PERMANENTLY FAILED after $MAX_ATTEMPTS attempts — moving on"
      overall_rc=1
      break
    fi
    echo "[gpu$GPU] seed $seed attempt $attempt/$MAX_ATTEMPTS starting"
    log="$LOGDIR/${CFG_TAG}_seed${seed}_gpu${GPU}_attempt${attempt}.log"
    timeout "$RUN_TIMEOUT" uv run python scripts/run.py "$CFG" --seed "$seed" >"$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && is_done "$seed"; then
      final=$(grep -o 'val_loss:[0-9.]*' "$log" | tail -1)
      echo "[gpu$GPU] seed $seed DONE (attempt $attempt, last $final)"
      break
    fi
    reason="rc=$rc"
    [ "$rc" -eq 124 ] && reason="HUNG (timeout ${RUN_TIMEOUT}s)"
    tailline=$(grep -iE "error|assert|Killed|OOM|CUDA" "$log" | tail -1 | cut -c1-160)
    backoff=${BACKOFFS[$((attempt - 1))]:-600}
    echo "[gpu$GPU] seed $seed attempt $attempt FAILED ($reason) ${tailline:+— $tailline }— retrying in ${backoff}s (checkpoint resume)"
    sleep "$backoff"
  done
done
echo "[gpu$GPU] babysitter finished (rc=$overall_rc)"
exit "$overall_rc"
