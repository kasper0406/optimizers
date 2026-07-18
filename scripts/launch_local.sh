#!/usr/bin/env bash
# Thin local launcher: bash scripts/launch_local.sh configs/<experiment>.yaml [extra args]
# Runs from anywhere; resolves the repo root from this script's location.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <config.yaml> [--seed N] [--out-dir DIR]" >&2
    exit 2
fi

cd "$REPO_ROOT"
exec uv run python scripts/run.py "$@"
