#!/usr/bin/env bash
# Hyperstack cloud launch path — HUMAN-EXECUTED (CLAUDE.md compute boundary,
# rule 4). This script does everything up to but NOT including provisioning:
# provisioning/deprovisioning the VM is a human-only step (console or MCP).
#
# Full walkthrough: docs/hyperstack-runbook.md
#
# Prereq: a provisioned Hyperstack GPU VM you can SSH into, with Docker + the
# NVIDIA container toolkit (Hyperstack GPU images ship both).
#
# Environment:
#   RM_VM          user@ip of the provisioned VM (required for remote commands)
#   RM_SSH_OPTS    extra ssh/rsync -e options (default: accept-new host keys)
#   RM_REMOTE_DIR  repo checkout dir on the VM   (default: ~/routed-muon)
#   RM_IMAGE       docker image tag              (default: routed-muon)
#   RM_STAGING     local staging dir for pulled results
#                  (default: <repo>/cloud_staging — NOT results/; results/ is
#                  append-only and only `ingest` moves finished files there)
#
# Commands (typical order):
#   push                 rsync the local repo to the VM (records local git SHA)
#   build                docker build on the VM (tags RM_GIT_SHA provenance)
#   run <config> [args]  run one experiment config in the container
#                        (args are forwarded to scripts/run.py, e.g. --seed 1000)
#   sweep <config>       expand + execute a sweep config in the container
#                        (seed policies 'eval'/'dev' resolve inside sweep.py;
#                        eval seeds 0-99 never appear in any config file)
#   pull                 rsync results JSONs from the VM into RM_STAGING
#   fill-cost <usd> <file...>
#                        set cost_usd on STAGED jsons (human reads the cost
#                        from Hyperstack billing; refuses files under results/)
#   ingest               validate staged jsons (cost_usd must be filled) and
#                        move them into results/ (append-only; never overwrites)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RM_VM="${RM_VM:-}"
RM_SSH_OPTS="${RM_SSH_OPTS:--o StrictHostKeyChecking=accept-new}"
RM_REMOTE_DIR="${RM_REMOTE_DIR:-\$HOME/routed-muon}"
RM_IMAGE="${RM_IMAGE:-routed-muon}"
RM_STAGING="${RM_STAGING:-$REPO_ROOT/cloud_staging}"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

need_vm() {
    if [[ -z "$RM_VM" ]]; then
        echo "error: set RM_VM=user@ip of the provisioned Hyperstack VM" >&2
        echo "(provisioning itself is human-only; see docs/hyperstack-runbook.md)" >&2
        exit 2
    fi
}

vm_ssh() {
    need_vm
    # shellcheck disable=SC2086
    ssh $RM_SSH_OPTS "$RM_VM" "$@"
}

vm_rsync() {
    need_vm
    # --info=stats1 only where supported (macOS ships openrsync without it)
    local info_flag=""
    rsync --info=stats1 --version >/dev/null 2>&1 && info_flag="--info=stats1"
    # shellcheck disable=SC2086
    rsync -az ${info_flag} -e "ssh $RM_SSH_OPTS" "$@"
}

local_git_sha() {
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown
}

cmd_push() {
    local sha
    sha="$(local_git_sha)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
        echo "WARNING: local tree is dirty; cloud results will record git_dirty" >&2
    fi
    vm_ssh "mkdir -p $RM_REMOTE_DIR $RM_REMOTE_DIR/results_out"
    vm_rsync --delete \
        --exclude .git --exclude .venv --exclude .pytest_cache \
        --exclude results --exclude cloud_staging --exclude sweeps \
        "$REPO_ROOT"/ "$RM_VM:$RM_REMOTE_DIR/"
    printf '%s\n' "$sha" | vm_ssh "cat > $RM_REMOTE_DIR/GIT_SHA"
    echo "pushed repo @ $sha -> $RM_VM:$RM_REMOTE_DIR"
}

cmd_build() {
    vm_ssh "cd $RM_REMOTE_DIR && docker build \
        --build-arg RM_GIT_SHA=\$(cat GIT_SHA 2>/dev/null || echo unknown) \
        -t $RM_IMAGE ."
}

cmd_run() {
    [[ $# -ge 1 ]] || usage
    local config="$1"; shift
    vm_ssh "docker run --rm --gpus all \
        -v $RM_REMOTE_DIR/results_out:/workspace/results \
        -v $RM_REMOTE_DIR/data:/workspace/data \
        $RM_IMAGE $config $*"
}

cmd_sweep() {
    [[ $# -ge 1 ]] || usage
    local config="$1"; shift
    # sweep.py resolves the seed policy at launch time (eval -> seeds 0-99 on
    # the run.py command line only) and executes every run inside the container.
    vm_ssh "docker run --rm --gpus all \
        -v $RM_REMOTE_DIR/results_out:/workspace/results \
        -v $RM_REMOTE_DIR/data:/workspace/data \
        --entrypoint uv $RM_IMAGE \
        run --frozen python scripts/sweep.py $config --out-dir /workspace/sweep_out --execute $*"
}

cmd_pull() {
    mkdir -p "$RM_STAGING"
    vm_rsync --ignore-existing "$RM_VM:$RM_REMOTE_DIR/results_out/" "$RM_STAGING/"
    echo "staged results in $RM_STAGING — next: fill-cost, then ingest"
}

cmd_fill_cost() {
    [[ $# -ge 2 ]] || usage
    (cd "$REPO_ROOT" && uv run python - "$@" <<'PY'
import json, sys
from pathlib import Path

usd = float(sys.argv[1])
repo_root = Path.cwd().resolve()
results_dir = (repo_root / "results").resolve()
for arg in sys.argv[2:]:
    path = Path(arg).resolve()
    if path == results_dir or results_dir in path.parents:
        raise SystemExit(
            f"refusing to edit {path}: results/ is append-only — fill costs on "
            "STAGED files (cloud_staging/) before ingest"
        )
    data = json.loads(path.read_text())
    data["cost_usd"] = usd
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"set cost_usd={usd} in {path}")
PY
    )
}

cmd_ingest() {
    (cd "$REPO_ROOT" && uv run python - "$RM_STAGING" <<'PY'
import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src import results_io

staging = Path(sys.argv[1])
files = sorted(staging.glob("*.json"))
if not files:
    raise SystemExit(f"no staged results JSONs in {staging}")
moved = 0
for path in files:
    result = results_io.load_result(path)  # schema validation
    if result["cost_usd"] is None and result["gpu_type"] not in ("cpu", "mps"):
        raise SystemExit(
            f"{path}: cost_usd is null for gpu_type {result['gpu_type']!r} — "
            "run fill-cost first (human reads Hyperstack billing)"
        )
    dest = results_io.RESULTS_DIR / path.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists; results/ is append-only")
    shutil.move(str(path), str(dest))
    print(f"ingested {dest.name}")
    moved += 1
print(f"ingested {moved} result file(s) into results/")
PY
    )
}

command="${1:-}"
shift || true
case "$command" in
    push)      cmd_push "$@" ;;
    build)     cmd_build "$@" ;;
    run)       cmd_run "$@" ;;
    sweep)     cmd_sweep "$@" ;;
    pull)      cmd_pull "$@" ;;
    fill-cost) cmd_fill_cost "$@" ;;
    ingest)    cmd_ingest "$@" ;;
    ""|-h|--help|help) usage ;;
    *) echo "unknown command: $command" >&2; usage ;;
esac
