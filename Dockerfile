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
    && apt-get install -y --no-install-recommends git curl ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13 forward-compat shim: uv.lock pins torch 2.13.0 (bundles CUDA 13.0
# runtime) but Hyperstack's newest image is R570 / CUDA 12.8 (verified
# 2026-08-02; R580 images no longer offered). cuda-compat-13-0 provides a
# CUDA-13 libcuda that runs on the 570 kernel driver (datacenter GPUs only —
# fine for A100/H100/L40 flavors). LD_LIBRARY_PATH makes the container prefer
# it over the toolkit-mounted host libcuda. No dependency pins change.
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-compat-13-0 \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}

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
