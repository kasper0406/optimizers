# Hyperstack image for Routed Muon cloud runs.
#
# Build (from repo root, human-launched):
#   docker build --build-arg RM_GIT_SHA=$(git rev-parse HEAD) -t routed-muon .
# Run any experiment config, mounting results/ for continuous sync:
#   docker run --gpus all -v /path/to/results:/workspace/results \
#       routed-muon configs/<experiment>.yaml [--seed N]
#
# CUDA-enabled PyTorch base; project deps are installed from uv.lock so the
# container runs the exact pinned versions (torch's linux wheels ship CUDA).

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (static binary)
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Dependency layer first for cache friendliness.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Project code (vendor submodules included when present in build context).
COPY . .
RUN uv sync --frozen --no-dev

# Git SHA provenance for runs inside the container (.git is not in the image);
# results_io.git_provenance() falls back to this env var.
ARG RM_GIT_SHA=unknown
ENV RM_GIT_SHA=${RM_GIT_SHA}

# Results are written here; mount a host/durable volume over it.
VOLUME ["/workspace/results"]

ENTRYPOINT ["uv", "run", "--frozen", "python", "scripts/run.py"]
CMD ["configs/smoke.yaml"]
