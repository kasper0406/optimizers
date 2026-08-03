"""Airbench smoke-run harness for the baseline zoo (WP0.4, part b).

Runs the vendored airbench94 recipe (vendor/airbench/airbench94_muon.py) with
the filter-parameter optimizer swapped for any zoo optimizer from
``src.optim.registry``. Model, data pipeline, augmentation, SGD side-optimizer
for biases/head, LR schedules, and evaluation (incl. TTA) are all the vendored
code / faithful ports of its ``main()`` (lines 340-432).

Requires CUDA: the vendored ``CifarLoader`` maps the dataset to "cuda"
(airbench94_muon.py:128) and the model runs in half precision. On the dev Mac
this module only needs to *import* and parse configs; the actual run happens
on a GPU box via::

    bash scripts/launch_local.sh configs/dev/airbench_smoke_muon.yaml
    # (once scripts/run.py registers this experiment -- see WIRING below), or
    uv run python -m src.optim.airbench_zoo configs/dev/airbench_smoke_muon.yaml

WIRING: scripts/run.py is owned by the WP0.0/WP0.1 tooling; to register this
experiment there, add::

    from src.optim import OPTIMIZER_REGISTRY as ZOO
    from src.optim.airbench_zoo import run_airbench_smoke
    OPTIMIZER_REGISTRY.update(ZOO)
    EXPERIMENT_REGISTRY["airbench_smoke"] = run_airbench_smoke

Until then, the ``python -m src.optim.airbench_zoo`` entrypoint below performs
the same registration at runtime (without editing scripts/run.py) and
delegates to ``scripts/run.py:main`` so results JSONs share the WP0.0 schema
and provenance fields.

torchvision note: the vendored script imports torchvision only for
``transforms.Normalize`` and the one-time CIFAR-10 download. torchvision is
not in pyproject.toml (locked; cannot be edited by WP0.4), so when it is
absent a minimal stand-in is installed into ``sys.modules`` before the vendor
import: Normalize as the standard (x - mean)/std, and a CIFAR10 class that
downloads/parses the canonical cifar-10-python.tar.gz with numpy only. With
real torchvision installed the stand-in is never used.
"""

from __future__ import annotations

import math
import pickle
import sys
import tarfile
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_AIRBENCH = REPO_ROOT / "vendor" / "airbench"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# ------------------------------------------------------- torchvision stand-in


class _Normalize:
    """transforms.Normalize equivalent for CHW tensors: (x - mean) / std."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = self.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - mean) / std


class _CIFAR10:
    """torchvision.datasets.CIFAR10 equivalent (data/targets/classes only).

    Downloads and parses the canonical CIFAR-10 python tarball with numpy;
    exposes exactly the attributes the vendored CifarLoader uses
    (airbench94_muon.py:122-126): .data (N,32,32,3 uint8), .targets, .classes.
    """

    def __init__(self, root, download: bool = False, train: bool = True):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        tar_path = root / "cifar-10-python.tar.gz"
        if not tar_path.exists():
            if not download:
                raise RuntimeError(f"CIFAR-10 not found at {tar_path}")
            urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        batch_names = (
            [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        )
        images, targets = [], []
        with tarfile.open(tar_path, "r:gz") as tar:
            for name in batch_names:
                with tar.extractfile(f"cifar-10-batches-py/{name}") as fh:
                    batch = pickle.load(fh, encoding="latin1")
                images.append(
                    np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
                )
                targets.extend(batch["labels"])
            with tar.extractfile("cifar-10-batches-py/batches.meta") as fh:
                meta = pickle.load(fh, encoding="latin1")
        self.data = np.concatenate(images).transpose(0, 2, 3, 1)  # NHWC uint8
        self.targets = list(targets)
        self.classes = list(meta["label_names"])


def _ensure_torchvision() -> None:
    """Install a minimal torchvision stand-in if the real one is missing."""
    try:
        import torchvision  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    datasets = types.ModuleType("torchvision.datasets")
    transforms.Normalize = _Normalize
    datasets.CIFAR10 = _CIFAR10
    tv.transforms = transforms
    tv.datasets = datasets
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.datasets"] = datasets


# ------------------------------------------------------------- vendor loading

_VENDOR_CACHE = None


def load_vendor_airbench():
    """Import vendor/airbench/airbench94_muon.py as a module (idempotent).

    Module-level code only defines the model/loader/eval helpers (training is
    under ``if __name__ == "__main__"``), so importing is side-effect free
    apart from reading sys.argv[0] for its self-logging feature.
    """
    global _VENDOR_CACHE
    if _VENDOR_CACHE is not None:
        return _VENDOR_CACHE
    _ensure_torchvision()
    import importlib.util

    path = VENDOR_AIRBENCH / "airbench94_muon.py"
    spec = importlib.util.spec_from_file_location("airbench94_muon_vendored", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The vendored script self-logs its own source via ``open(sys.argv[0])`` at
    # import time (airbench94_muon.py:14). Point argv[0] at the vendored file
    # during import so (a) the import survives contexts where argv[0] is not a
    # readable file (e.g. ``python -c``) and (b) the logged code is the actual
    # vendored source, matching the reference's intent.
    argv0 = sys.argv[0]
    sys.argv[0] = str(path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv[0] = argv0
    _VENDOR_CACHE = module
    return module


# ------------------------------------------------- batch-sampling ablation

VALID_SAMPLING = (None, "with_replacement")

# T1 EMA-as-anneal experiment (docs/litreview/j-theory-theorem-sweep.md §5):
# 'linear' is the vendored schedule (LR decays linearly to zero over the run,
# airbench94_muon.py:377-379); 'constant' holds every group except the
# whiten bias at its initial LR for the whole run (the whiten bias keeps its
# stock 3-epoch decay in both arms — it is grad-disabled afterwards and is
# not part of the schedule comparison).
VALID_LR_SCHEDULES = ("linear", "constant")


def _resolve_lr_schedule(recipe_cfg: Dict[str, Any]) -> str:
    schedule = recipe_cfg.get("lr_schedule", "linear")
    if schedule not in VALID_LR_SCHEDULES:
        raise SystemExit(
            f"recipe.lr_schedule must be one of {VALID_LR_SCHEDULES}, "
            f"got {schedule!r}"
        )
    return schedule


def _resolve_ema(recipe_cfg: Dict[str, Any]):
    """Validate the optional recipe.ema block; returns the gamma list or None.

    CPU-safe (config validation only). Shape: ``ema: {gammas: [0.9, ...]}``.
    """
    ema_cfg = recipe_cfg.get("ema", None)
    if ema_cfg is None:
        return None
    from src.optim.ema_weights import validate_gammas

    if not isinstance(ema_cfg, dict) or set(ema_cfg) != {"gammas"}:
        raise SystemExit(
            "recipe.ema must be a mapping with exactly the key 'gammas', "
            f"got {ema_cfg!r}"
        )
    return validate_gammas(ema_cfg["gammas"])


def _resolve_sampling(recipe_cfg: Dict[str, Any]):
    """Validate recipe.sampling.

    ``None`` (key absent) = the vendored CifarLoader behavior, bit-identical
    to the reference: an epoch is a random permutation of the training set,
    partitioned into batches (sampling WITHOUT replacement within an epoch).

    ``"with_replacement"`` = the Phase-1 disambiguation ablation: each
    training step draws its ``batch_size`` indices i.i.d. WITH replacement
    from the training set (see :func:`iter_batches_with_replacement`).
    """
    sampling = recipe_cfg.get("sampling", None)
    if sampling not in VALID_SAMPLING:
        raise SystemExit(
            f"recipe.sampling must be one of {VALID_SAMPLING}, got {sampling!r}"
        )
    return sampling


def iter_batches_with_replacement(loader, ab):
    """One epoch over the vendored CifarLoader with i.i.d. WITH-replacement
    batch index draws (Phase-1 sampling ablation).

    Faithful replica of ``CifarLoader.__iter__`` (airbench94_muon.py:148-173)
    -- identical epoch-0 preprocessing cache (normalize -> pre-flip -> reflect
    pad), identical per-epoch random crop of the whole padded set, identical
    deterministic every-other-epoch full flip, identical number of batches
    per epoch -- EXCEPT the final index generation: the reference partitions
    one ``torch.randperm`` (each image exactly once per epoch); this draws
    each batch's indices via ``torch.randint`` (i.i.d. with replacement).

    Documented deviations from the vendored path (and the only ones):
    1. An image may appear 0 or several times per epoch, even within one
       batch; repeated draws within an epoch share the SAME augmentation
       realization (crop/flip are materialized once per epoch for the whole
       set, exactly as in the reference, and then indexed).
    2. The index RNG consumes the device RNG stream via ``torch.randint``
       (one call per batch) instead of one ``torch.randperm`` per epoch, so
       downstream RNG draws differ from the reference stream (inherent to
       any sampling change).
    """
    if loader.epoch == 0:
        images = loader.proc_images["norm"] = loader.normalize(loader.images)
        if loader.aug.get("flip", False):
            images = loader.proc_images["flip"] = ab.batch_flip_lr(images)
        pad = loader.aug.get("translate", 0)
        if pad > 0:
            loader.proc_images["pad"] = torch.nn.functional.pad(
                images, (pad,) * 4, "reflect"
            )
    if loader.aug.get("translate", 0) > 0:
        images = ab.batch_crop(loader.proc_images["pad"], loader.images.shape[-2])
    elif loader.aug.get("flip", False):
        images = loader.proc_images["flip"]
    else:
        images = loader.proc_images["norm"]
    if loader.aug.get("flip", False):
        if loader.epoch % 2 == 1:
            images = images.flip(-1)

    loader.epoch += 1

    n = len(images)
    for _ in range(len(loader)):
        idxs = torch.randint(n, (loader.batch_size,), device=images.device)
        yield (images[idxs], loader.labels[idxs])


class WithReplacementLoader:
    """Iteration wrapper engaging :func:`iter_batches_with_replacement`.

    Only constructed when ``recipe.sampling: with_replacement`` is set; the
    default path never touches this class and keeps the vendored loader's
    iterator bit-identical to the reference.
    """

    def __init__(self, loader, ab):
        self.loader = loader
        self._ab = ab

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter_batches_with_replacement(self.loader, self._ab)

    def __getattr__(self, name):
        # Delegate everything else (normalize, images, ...) to the vendored
        # loader so the harness's whiten-init path works unchanged.
        return getattr(self.loader, name)


# ----------------------------------------------------------------- experiment

# Routing-telemetry time-series cadence (Gate-1 amendment A5): every N steps
# the aggregate last-step stats are appended to metrics["routing_timeseries"].
ROUTING_TS_EVERY = 10


def run_airbench_smoke(
    config: Dict[str, Any],
    device: torch.device,
    _hub_factory=None,
    _batch_hook=None,
    _pre_step_hook=None,
    _post_step_hook=None,
) -> Dict[str, Any]:
    """One airbench94 training run with a zoo optimizer on the filter params.

    Faithful port of vendor/airbench/airbench94_muon.py:main() (lines
    340-432) with:
    - optimizer2 (filter params) built from src.optim.registry per config;
    - the reference's per-step weight renormalization (line 83, inside the
      vendored Muon.step) applied harness-side to filter params so every zoo
      optimizer trains under the identical recipe (config-switchable);
    - torch.compile optional (smoke default: off);
    - wall-clock timing via CUDA events as in the reference.

    ``_hub_factory`` (internal; used by :func:`run_airbench_instrumented`):
    callable ``(model, optimizer2, filter_params) -> InstrumentationHub``.
    When given, the hub observes every step -- ``capture_grads()`` right
    before the optimizer steps (the raw PRE-momentum gradient; the vendored
    Muon's ``step()`` must never be assumed to leave ``p.grad`` intact) and
    ``after_step()`` right after, before ``zero_grad``.  Instrumentation is
    strictly read-only: it never modifies parameters, gradients, optimizer
    state, or any update.

    ``_batch_hook`` (internal; used by the HVP-enabled instrumented runs):
    callable ``(inputs, labels)`` invoked once per training step with the
    current (augmented, normalized) batch, right before ``capture_grads()``.
    ``None`` (default) leaves the loop untouched.

    ``_pre_step_hook`` / ``_post_step_hook`` (internal; used by the
    directional-smoothness probe): ``_pre_step_hook(step)`` runs immediately
    before ``optimizer.step()`` (gradients present, weights still pre-update)
    and ``_post_step_hook(step, lr)`` immediately after it, with ``lr`` the
    filter-parameter learning rate actually applied on that step.  ``step`` is
    1-based, matching the instrumentation hub's step counter.  Both are
    read-only observers; ``None`` (default) leaves the loop untouched.
    """
    from src.optim.registry import build_optimizer

    # Config validation first (CPU-safe): recipe.sampling gates the Phase-1
    # with-replacement ablation; absent = vendored behavior, bit-identical.
    # recipe.lr_schedule / recipe.ema gate the T1 EMA-as-anneal arms; defaults
    # (linear, no EMA) keep the vendored behavior bit-identical.
    sampling = _resolve_sampling(config.get("recipe", {}))
    lr_schedule = _resolve_lr_schedule(config.get("recipe", {}))
    ema_gammas = _resolve_ema(config.get("recipe", {}))

    if device.type != "cuda":
        raise SystemExit(
            "airbench_smoke requires a CUDA device: the vendored CifarLoader "
            "maps data to cuda (airbench94_muon.py:128) and the model is half "
            "precision. Run this config on a GPU box."
        )

    ab = load_vendor_airbench()

    opt_cfg = dict(config.get("optimizer", {}))
    opt_name = opt_cfg.pop("name")
    train_cfg = config.get("train", {})
    recipe_cfg = config.get("recipe", {})
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))

    epochs = float(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 2000))
    bias_lr = float(recipe_cfg.get("bias_lr", 0.053))
    head_lr = float(recipe_cfg.get("head_lr", 0.67))
    wd = float(recipe_cfg.get("sgd_weight_decay", 2e-6)) * batch_size
    normalize_filter_weights = bool(recipe_cfg.get("normalize_filter_weights", True))
    tta_level = int(recipe_cfg.get("tta_level", 2))

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if bool(recipe_cfg.get("compile", False)):
        model.compile()

    test_loader = ab.CifarLoader(data_root, train=False, batch_size=2000)
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    if sampling == "with_replacement":
        # Phase-1 sampling ablation; len() and per-epoch augmentation are
        # identical to the vendored loader, only the index draw changes.
        train_loader = WithReplacementLoader(train_loader, ab)
    total_train_steps = math.ceil(epochs * len(train_loader))
    whiten_bias_train_steps = min(
        math.ceil(3 * len(train_loader)), total_train_steps
    )

    # Parameter split identical to the reference (lines 356-361)
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=(device.type == "cuda")
    )
    if opt_name == "vendor_muon":
        # WP0.1 baseline: the vendored Muon itself (airbench94_muon.py:56-84).
        # It renormalizes filter weights inside step() (line 83), so the
        # harness-side renormalization must stay off to avoid applying it twice.
        if normalize_filter_weights:
            raise SystemExit(
                "optimizer 'vendor_muon' renormalizes weights inside step(); "
                "set recipe.normalize_filter_weights: false"
            )
        optimizer2 = ab.Muon(filter_params, **opt_cfg)
    else:
        optimizer2 = build_optimizer(opt_name, filter_params, opt_cfg)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    hub = None
    if _hub_factory is not None:
        hub = _hub_factory(model, optimizer2, filter_params)

    # Routing telemetry (Gate-1 amendment A5): optimizers exposing
    # routing_stats() (RoutedMuon) get their per-channel occupancy / treated
    # fraction / gain distribution recorded -- full dict at end of run plus a
    # coarse aggregate time series every ROUTING_TS_EVERY steps. Read-only.
    track_routing = hasattr(optimizer2, "routing_stats")
    routing_timeseries = []

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    ema = None
    ema_val_accs: Dict[str, list] = {}
    if ema_gammas is not None:
        from src.optim.ema_weights import WeightEMA

        # Track every parameter plus the float buffers (BatchNorm running
        # stats) so an EMA checkpoint is a complete, evaluable model state.
        # Constructed after init_whiten so shadows start from the real init.
        tracked = list(model.named_parameters()) + [
            (name, buf)
            for name, buf in model.named_buffers()
            if buf.is_floating_point()
        ]
        ema = WeightEMA(tracked, ema_gammas)
        ema_val_accs = {str(g): [] for g in ema.gammas}

    val_accs = []
    train_acc = float("nan")
    for _epoch in range(math.ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            torch.nn.functional.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="sum"
            ).backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (
                    1 - step / whiten_bias_train_steps
                )
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                if lr_schedule == "constant":
                    group["lr"] = group["initial_lr"]
                else:
                    group["lr"] = group["initial_lr"] * (
                        1 - step / total_train_steps
                    )
            if normalize_filter_weights:
                # airbench94_muon.py:83 (recipe step, applied uniformly)
                for p in filter_params:
                    p.data.mul_(len(p.data) ** 0.5 / p.data.norm())
            if _batch_hook is not None:
                _batch_hook(inputs, labels)  # current batch for HVP probes
            if hub is not None:
                hub.capture_grads()  # raw pre-momentum G, before any step()
            if _pre_step_hook is not None:
                _pre_step_hook(step + 1)  # 1-based, as the hub counts
            step_lr = optimizer2.param_groups[0]["lr"]
            for opt in optimizers:
                opt.step()
            if _post_step_hook is not None:
                _post_step_hook(step + 1, step_lr)
            if hub is not None:
                hub.after_step()  # reads captured G + post-step momentum
            if ema is not None:
                ema.update()  # post-step, post-renorm weights
            model.zero_grad(set_to_none=True)
            step += 1
            if track_routing and step % ROUTING_TS_EVERY == 0:
                agg = optimizer2.routing_stats()["aggregate"]["last"]
                if agg is not None:
                    routing_timeseries.append(
                        {
                            "step": step,
                            "treated_fraction": agg["treated_fraction"],
                            "n_signal": agg["n_signal"],
                            "n_noise": agg["n_noise"],
                            "n_oscillating": agg["n_oscillating"],
                            "n_treated": agg["n_treated"],
                            "n_in_confidence_window": agg[
                                "n_in_confidence_window"
                            ],
                        }
                    )
            if step >= total_train_steps:
                break
        stop_timer()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_accs.append(ab.evaluate(model, test_loader, tta_level=0))
        if ema is not None:
            # Per-epoch EMA readout (outside the timed region, like val_accs):
            # swap each shadow in, evaluate, restore the live training state
            # bit-exactly. TTA off here for speed; final TTA per gamma below.
            for g in ema.gammas:
                with ema.applied(g):
                    ema_val_accs[str(g)].append(
                        ab.evaluate(model, test_loader, tta_level=0)
                    )
        if step >= total_train_steps:
            break

    start_timer()
    tta_val_acc = (
        ab.evaluate(model, test_loader, tta_level=tta_level) if tta_level else None
    )
    stop_timer()

    ema_tta_val_accs = None
    if ema is not None:
        ema_tta_val_accs = {}
        for g in ema.gammas:
            with ema.applied(g):
                ema_tta_val_accs[str(g)] = (
                    ab.evaluate(model, test_loader, tta_level=tta_level)
                    if tta_level
                    else None
                )

    metrics = {
        "optimizer": opt_name,
        "epochs": epochs,
        "steps": step,
        "train_acc_last": train_acc,
        "val_accs": val_accs,
        "val_acc": val_accs[-1],
        "tta_val_acc": tta_val_acc,
        "time_seconds": time_seconds,
    }
    if sampling is not None:
        metrics["sampling"] = sampling  # ablation provenance; absent = vendor
    if lr_schedule != "linear" or ema is not None:
        # T1 arm provenance; keys absent on stock runs so pre-T1 outputs are
        # byte-identical.
        metrics["lr_schedule"] = lr_schedule
    if ema is not None:
        metrics["ema_gammas"] = list(ema.gammas)
        metrics["ema_val_accs"] = ema_val_accs
        metrics["ema_tta_val_accs"] = ema_tta_val_accs
    if track_routing:
        # Gate-1 amendment A5: end-of-run routing telemetry + coarse series.
        metrics["routing_stats"] = optimizer2.routing_stats()
        metrics["routing_timeseries"] = routing_timeseries
    return metrics


def run_airbench(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """WP0.1 baseline: the stock vendored airbench94 recipe, unmodified.

    Same harness as :func:`run_airbench_smoke` but pinned to the vendored
    Muon at the record hyperparameters (airbench94_muon.py:362 —
    lr=0.24, momentum=0.6, nesterov=True) with torch.compile on, exactly as
    the reference ``main()``. Config may not override the optimizer: the
    point of WP0.1 is the untouched reference distribution.
    """
    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench' is the stock WP0.1 baseline; it does not "
            "accept an optimizer override (use experiment 'airbench_smoke')."
        )
    merged = dict(config)
    merged["optimizer"] = dict(
        name="vendor_muon", lr=0.24, momentum=0.6, nesterov=True
    )
    recipe = dict(config.get("recipe", {}))
    recipe["normalize_filter_weights"] = False  # vendored Muon.step does it
    recipe.setdefault("compile", True)  # the reference compiles the model
    merged["recipe"] = recipe
    return run_airbench_smoke(merged, device)


def run_airbench_ema(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """T1 EMA-as-anytime-anneal arms (docs/litreview/j-theory-theorem-sweep.md §5).

    The stock WP0.1 recipe (vendored Muon at record hyperparameters, compile
    on — identical pinning to :func:`run_airbench`) plus a mandatory
    ``recipe.ema`` block (weight-EMA readouts per epoch and at final TTA) and
    an optional ``recipe.lr_schedule`` arm switch (``linear`` = vendored
    baseline arm, ``constant`` = schedule-free arm whose EMA readout is the
    anneal substitute). Training trajectories are unaffected by the EMA
    machinery: shadows are read-only observers and eval swaps restore the
    live state bit-exactly, so the ``linear`` arm's trajectory is the stock
    baseline trajectory.

    Dev-seed measurement experiment; never a comparison-table entry.
    """
    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench_ema' pins the stock WP0.1 recipe; it does "
            "not accept an optimizer override (use experiment 'airbench_smoke')."
        )
    if _resolve_ema(config.get("recipe", {})) is None:
        raise SystemExit(
            "experiment 'airbench_ema' requires a recipe.ema block "
            "(e.g. ema: {gammas: [0.9, 0.96, 0.99]}); for stock runs without "
            "EMA use experiment 'airbench'."
        )
    merged = dict(config)
    merged["optimizer"] = dict(
        name="vendor_muon", lr=0.24, momentum=0.6, nesterov=True
    )
    recipe = dict(config.get("recipe", {}))
    recipe["normalize_filter_weights"] = False  # vendored Muon.step does it
    recipe.setdefault("compile", True)  # the reference compiles the model
    merged["recipe"] = recipe
    return run_airbench_smoke(merged, device)


def _resolve_branch(config: Dict[str, Any], total_train_steps: int = None):
    """Validate the ``branch:`` block of an anneal-dissection config.

    Returns (branch_steps, anneal_lengths). CPU-safe. When total_train_steps
    is given, branch points must lie in (0, total]."""
    branch = config.get("branch")
    if not isinstance(branch, dict) or set(branch) != {"branch_steps", "anneal_lengths"}:
        raise SystemExit(
            "experiment 'airbench_anneal_branch' needs a branch: block with "
            "exactly the keys branch_steps and anneal_lengths"
        )
    steps = branch["branch_steps"]
    lengths = branch["anneal_lengths"]
    for name, values in (("branch_steps", steps), ("anneal_lengths", lengths)):
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in values)
            or sorted(set(values)) != values
        ):
            raise SystemExit(
                f"branch.{name} must be a non-empty strictly-increasing list "
                f"of ints, got {values!r}"
            )
    if steps[0] < 1:
        raise SystemExit("branch.branch_steps must be >= 1")
    if lengths[0] < 0:
        raise SystemExit("branch.anneal_lengths must be >= 0")
    if total_train_steps is not None and steps[-1] > total_train_steps:
        raise SystemExit(
            f"branch.branch_steps beyond the run: {steps[-1]} > {total_train_steps}"
        )
    return steps, lengths


def run_airbench_anneal_branch(
    config: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Anneal dissection ("how short is the last mile?") — flow-first program
    item 1, docs/litreview/j-theory-theorem-sweep.md §6.

    Stock airbench94 recipe (vendored Muon at record hyperparameters) trained
    at CONSTANT LR (the T1 constant-arm schedule: every group pinned at its
    initial LR; the whiten bias keeps its stock early decay and is
    grad-disabled after 3 epochs). At each configured branch step the live
    training state is snapshotted and, for each anneal length k, a branch
    anneals the LR linearly to zero over k steps — all branches at one branch
    point share the snapshot AND the identical cached continuation batches,
    so accuracy differences are attributable to the anneal length alone. The
    base trajectory then resumes from the snapshot through those same cached
    batches, unperturbed by the branches (bit-exact restore).

    Readout: accuracy vs k per branch point. The saturation k* measures how
    much of the anneal is fast dynamical relaxation; paired stock-schedule
    finals for the same dev seeds live in the T1 results
    (`results/airbench_ema_*`, lr_schedule=linear).

    Dev-seed measurement experiment; never a comparison-table entry.
    """
    from src.optim.train_snapshot import (
        restore_training_state,
        snapshot_training_state,
    )

    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench_anneal_branch' pins the stock recipe; it "
            "does not accept an optimizer override"
        )
    branch_steps, anneal_lengths = _resolve_branch(config)
    sampling = _resolve_sampling(config.get("recipe", {}))
    if sampling is not None:
        raise SystemExit("airbench_anneal_branch supports vendored sampling only")

    if device.type != "cuda":
        raise SystemExit("airbench_anneal_branch requires a CUDA device")

    ab = load_vendor_airbench()

    train_cfg = config.get("train", {})
    recipe_cfg = config.get("recipe", {})
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))
    epochs = float(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 2000))
    bias_lr = float(recipe_cfg.get("bias_lr", 0.053))
    head_lr = float(recipe_cfg.get("head_lr", 0.67))
    wd = float(recipe_cfg.get("sgd_weight_decay", 2e-6)) * batch_size
    tta_level = int(recipe_cfg.get("tta_level", 2))

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if bool(recipe_cfg.get("compile", True)):
        model.compile()

    test_loader = ab.CifarLoader(data_root, train=False, batch_size=2000)
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    total_train_steps = math.ceil(epochs * len(train_loader))
    whiten_bias_train_steps = min(math.ceil(3 * len(train_loader)), total_train_steps)
    _resolve_branch(config, total_train_steps)  # now with the bound known

    # Parameter split + optimizers exactly as the stock harness
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    optimizer2 = ab.Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    def batch_stream():
        while True:
            for batch in train_loader:
                yield batch

    stream = batch_stream()

    def train_step(inputs, labels, step_index, lr_scale):
        """One training step at global step_index; lr_scale multiplies every
        non-whiten group's initial LR (1.0 = the constant base schedule)."""
        # ab.evaluate() leaves the model in eval mode; the stock loop restores
        # train mode once per epoch, but here evals happen mid-stream at every
        # branch, so re-assert it per step (BatchNorm must train in train mode).
        model.train()
        outputs = model(
            inputs, whiten_bias_grad=(step_index < whiten_bias_train_steps)
        )
        torch.nn.functional.cross_entropy(
            outputs, labels, label_smoothing=0.2, reduction="sum"
        ).backward()
        for group in optimizer1.param_groups[:1]:
            group["lr"] = group["initial_lr"] * (
                1 - step_index / whiten_bias_train_steps
            )
        for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    k_max = anneal_lengths[-1]
    base_val_accs: Dict[str, float] = {}
    branches: Dict[str, Dict[str, Dict[str, float]]] = {}
    pending: list = []  # cached continuation batches the base must consume
    remaining_branches = list(branch_steps)

    step = 0
    start_timer()
    while True:
        if remaining_branches and step == remaining_branches[0]:
            t_b = remaining_branches.pop(0)
            stop_timer()
            # Cache the continuation batches once; branches and the resumed
            # base all consume this identical stream.
            cache = [
                pending.pop(0) if pending else next(stream) for _ in range(k_max)
            ]
            snap = snapshot_training_state(model, optimizers)
            base_val_accs[str(t_b)] = ab.evaluate(model, test_loader, tta_level=0)
            branches[str(t_b)] = {}
            start_timer()
            for k in anneal_lengths:
                restore_training_state(model, optimizers, snap)
                for i in range(k):
                    inputs, labels = cache[i]
                    train_step(inputs, labels, t_b + i, (k - i) / k)
                stop_timer()
                entry = {"val_acc": ab.evaluate(model, test_loader, tta_level=0)}
                entry["tta_val_acc"] = (
                    ab.evaluate(model, test_loader, tta_level=tta_level)
                    if tta_level
                    else None
                )
                branches[str(t_b)][str(k)] = entry
                start_timer()
            restore_training_state(model, optimizers, snap)
            pending = cache  # base resumes through the same batches
        if step >= total_train_steps:
            break
        inputs, labels = pending.pop(0) if pending else next(stream)
        train_step(inputs, labels, step, 1.0)
        step += 1
    stop_timer()

    final_val_acc = ab.evaluate(model, test_loader, tta_level=0)
    final_tta_val_acc = (
        ab.evaluate(model, test_loader, tta_level=tta_level) if tta_level else None
    )

    return {
        "optimizer": "vendor_muon",
        "epochs": epochs,
        "steps": step,
        "lr_schedule": "constant",
        "branch_steps": branch_steps,
        "anneal_lengths": anneal_lengths,
        "base_val_accs": base_val_accs,
        "branches": branches,
        "final_val_acc": final_val_acc,
        "final_tta_val_acc": final_tta_val_acc,
        "time_seconds": time_seconds,
    }


# ------------------------------------------------- teleportation go/no-go gate

# Keys the ``gate:`` block must carry, exactly.
GATE_KEYS = ("snapshot_steps", "n_samples", "spread", "refine_iters", "probe_batches")

# A proposal counts as ON the loss level set iff |(L - L0) / L0| is below this.
# Not a science threshold: it is the numerical definition of "same loss" for
# the constrained search, and every draw's realized rel_dloss is reported so
# the invariance itself is auditable from the results JSON.
TELEPORT_INVARIANCE_TOL = 1e-3

# Log-scale proposal standard deviation for the refinement random search.
TELEPORT_REFINE_STEP = 0.25


def _resolve_gate(config: Dict[str, Any], total_train_steps: int = None):
    """Validate the ``gate:`` block of a teleportation-gate config.

    Returns the normalized gate dict. CPU-safe (runs before any CUDA work).
    When ``total_train_steps`` is given, snapshot points must lie in
    (0, total].
    """
    gate = config.get("gate")
    if not isinstance(gate, dict) or set(gate) != set(GATE_KEYS):
        raise SystemExit(
            "experiment 'airbench_teleport_gate' needs a gate: block with "
            f"exactly the keys {', '.join(GATE_KEYS)}"
        )
    steps = gate["snapshot_steps"]
    if (
        not isinstance(steps, list)
        or not steps
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in steps)
        or sorted(set(steps)) != steps
    ):
        raise SystemExit(
            "gate.snapshot_steps must be a non-empty strictly-increasing list "
            f"of ints, got {steps!r}"
        )
    if steps[0] < 1:
        raise SystemExit("gate.snapshot_steps must be >= 1")
    counts = {}
    for name, minimum in (
        ("n_samples", 1),
        ("refine_iters", 0),
        ("probe_batches", 1),
    ):
        value = gate[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise SystemExit(f"gate.{name} must be an int >= {minimum}, got {value!r}")
        counts[name] = value
    spread = gate["spread"]
    if isinstance(spread, bool) or not isinstance(spread, (int, float)):
        raise SystemExit(f"gate.spread must be a number > 1, got {spread!r}")
    spread = float(spread)
    if not spread > 1.0:
        raise SystemExit(f"gate.spread must be > 1, got {spread}")
    if total_train_steps is not None and steps[-1] > total_train_steps:
        raise SystemExit(
            f"gate.snapshot_steps beyond the run: {steps[-1]} > {total_train_steps}"
        )
    return {"snapshot_steps": list(steps), "spread": spread, **counts}


def _ratio(value: float, base: float) -> float:
    """value/base, or NaN when the base is zero (kept out of the JSON math)."""
    return float(value / base) if base else float("nan")


def run_airbench_teleport_gate(
    config: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Teleportation go/no-go gate — flow-first program item 4,
    docs/litreview/j-theory-theorem-sweep.md section 6 (T6).

    Symmetry teleportation provably accelerates iff the gradient norm VARIES
    along the loss level set.  The conv->BatchNorm channel rescalings are a
    closed-form, loss-invariant symmetry orbit of this network
    (:mod:`src.instrument.orbit`), so the variation is directly measurable.
    This experiment measures it at representative training states.

    Stock airbench94 recipe (vendored Muon at the record hyperparameters
    lr=0.24 momentum=0.6 nesterov, compile per recipe) trained under the STOCK
    LINEAR LR decay -- unlike the anneal-dissection experiment, this one wants
    the states the real recipe actually visits.  At each configured snapshot
    step the live training state is snapshotted, ``probe_batches`` batches are
    cached off the stream, and on that fixed probe loss (cross entropy,
    label_smoothing 0.2, reduction sum -- the training loss) the harness:

      1. records the base loss L0 and base gradient norms;
      2. draws ``n_samples`` log-uniform channel rescalings in
         [1/spread, spread], and for each records the realized relative loss
         change and the Euclidean / Frobenius / NUCLEAR gradient-norm ratios;
      3. runs a ``refine_iters``-iteration random search that MAXIMIZES the
         nuclear-norm sum subject to |rel_dloss| < TELEPORT_INVARIANCE_TOL,
         starting from the best feasible random draw.

    Every probe evaluation is undone by a bit-exact restore from
    ``src.optim.train_snapshot`` (multiplicative inversion is lossy in half
    precision), and the cached probe batches are fed back to the base
    trajectory afterwards, so the base run is unperturbed.

    Reports BOTH Euclidean (Frobenius) and nuclear norms: the nuclear norm is
    the Muon-relevant one (a Muon step's first-order loss decrease is
    ``<G, polar(G)> = ||G||_*``), the Euclidean one is what the teleportation
    literature states its conditions in.

    The experiment measures and reports only.  It evaluates no gate and
    carries no success threshold; the kill criterion (O(1%) heterogeneity)
    lives in the config header and is judged by a human.

    Dev-seed measurement experiment; never a comparison-table entry.
    """
    from src.instrument.orbit import (
        apply_channel_scales,
        find_conv_bn_pairs,
        grad_norms,
        pair_names,
        refine_scales,
        sample_log_uniform_scales,
    )
    from src.optim.train_snapshot import (
        restore_training_state,
        snapshot_training_state,
    )

    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench_teleport_gate' pins the stock recipe; it "
            "does not accept an optimizer override"
        )
    gate = _resolve_gate(config)
    sampling = _resolve_sampling(config.get("recipe", {}))
    if sampling is not None:
        raise SystemExit("airbench_teleport_gate supports vendored sampling only")

    if device.type != "cuda":
        raise SystemExit("airbench_teleport_gate requires a CUDA device")

    ab = load_vendor_airbench()

    train_cfg = config.get("train", {})
    recipe_cfg = config.get("recipe", {})
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))
    epochs = float(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 2000))
    bias_lr = float(recipe_cfg.get("bias_lr", 0.053))
    head_lr = float(recipe_cfg.get("head_lr", 0.67))
    wd = float(recipe_cfg.get("sgd_weight_decay", 2e-6)) * batch_size
    tta_level = int(recipe_cfg.get("tta_level", 0))

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if bool(recipe_cfg.get("compile", True)):
        model.compile()

    test_loader = ab.CifarLoader(data_root, train=False, batch_size=2000)
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    total_train_steps = math.ceil(epochs * len(train_loader))
    whiten_bias_train_steps = min(math.ceil(3 * len(train_loader)), total_train_steps)
    _resolve_gate(config, total_train_steps)  # now with the bound known

    # Parameter split + optimizers exactly as the stock harness
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    optimizer2 = ab.Muon(
        filter_params, lr=AIRBENCH_STOCK_LR, momentum=0.6, nesterov=True
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    pairs = find_conv_bn_pairs(model)
    if not pairs:
        raise SystemExit("no conv->BatchNorm pairs found; nothing to teleport along")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    # Orbit draws use their own CPU generator, seeded from the (already
    # run-seeded) global CPU RNG *after* model init so the init stream is
    # untouched. Data augmentation draws from the CUDA RNG, so the probe's
    # sampling cannot perturb the training trajectory at all.
    gate_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator().manual_seed(gate_seed)

    def batch_stream():
        while True:
            for batch in train_loader:
                yield batch

    stream = batch_stream()

    def train_step(inputs, labels, step_index):
        """One stock-schedule training step (linear LR decay to zero)."""
        # Probes leave the model in train mode already, but ab.evaluate() at the
        # end of the run does not; re-assert per step as the branch harness does.
        model.train()
        outputs = model(
            inputs, whiten_bias_grad=(step_index < whiten_bias_train_steps)
        )
        torch.nn.functional.cross_entropy(
            outputs, labels, label_smoothing=0.2, reduction="sum"
        ).backward()
        for group in optimizer1.param_groups[:1]:
            group["lr"] = group["initial_lr"] * (
                1 - step_index / whiten_bias_train_steps
            )
        for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
            group["lr"] = group["initial_lr"] * (1 - step_index / total_train_steps)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    def teleport_probe(step_index, cache):
        """Measure gradient-size variation along the orbit at this state."""
        snap = snapshot_training_state(model, optimizers)
        whiten_grad = step_index < whiten_bias_train_steps

        def measure():
            """Loss + gradient norms on the fixed probe batches; grads zeroed.

            The logits are cast to float32 first.  The training loop leaves
            them half, but a half-precision SUM-reduced loss over these batch
            sizes lands near 1e4, where fp16 spacing is ~4 -- a relative
            resolution of ~7e-4, which would swamp an invariance measurement
            whose whole point is that |rel_dloss| is small.  The cast applies
            identically to the base point and every orbit draw.  (The residual
            floor is fp16 rounding of the RESCALED weights themselves, which
            moves the point slightly off the exact orbit; the reported
            ``max_abs_rel_dloss`` is exactly that floor.)
            """
            model.train()
            total = None
            for inputs, labels in cache:
                outputs = model(inputs, whiten_bias_grad=whiten_grad)
                part = torch.nn.functional.cross_entropy(
                    outputs.float(), labels, label_smoothing=0.2, reduction="sum"
                )
                total = part if total is None else total + part
            total.backward()
            value = float(total.detach().float().item())
            norms = grad_norms(model, pairs)
            model.zero_grad(set_to_none=True)
            return value, norms

        base_loss, base = measure()
        restore_training_state(model, optimizers, snap)

        def evaluate(scales):
            apply_channel_scales(pairs, scales)
            loss_value, norms = measure()
            restore_training_state(model, optimizers, snap)
            return {
                "rel_dloss": _ratio(loss_value - base_loss, abs(base_loss)),
                "total_grad_ratio": _ratio(norms["total"], base["total"]),
                "nuclear_sum_ratio": _ratio(norms["nuclear_sum"], base["nuclear_sum"]),
                "fro_ratio": [
                    _ratio(v, b) for v, b in zip(norms["fro"], base["fro"])
                ],
                "nuc_ratio": [
                    _ratio(v, b) for v, b in zip(norms["nuclear"], base["nuclear"])
                ],
            }

        draws = []
        for _ in range(gate["n_samples"]):
            scales = sample_log_uniform_scales(pairs, gate["spread"], generator)
            draws.append((scales, evaluate(scales)))

        feasible = [
            (s, r)
            for s, r in draws
            if abs(r["rel_dloss"]) < TELEPORT_INVARIANCE_TOL
            and math.isfinite(r["nuclear_sum_ratio"])
        ]
        best_random = None
        init_scales = [torch.ones(int(c.weight.shape[0])) for c, _b, _n in pairs]
        if feasible:
            scales, best_random = max(
                feasible, key=lambda sr: sr[1]["nuclear_sum_ratio"]
            )
            init_scales = scales

        # Constrained refinement: maximize the nuclear-norm sum, discarding any
        # proposal that leaves the level set.
        best_holder: Dict[str, Any] = {"value": float("-inf"), "record": None}

        def objective(scales):
            record = evaluate(scales)
            if abs(record["rel_dloss"]) >= TELEPORT_INVARIANCE_TOL:
                return None
            value = record["nuclear_sum_ratio"]
            if not math.isfinite(value):
                return None
            if value > best_holder["value"]:
                best_holder["value"] = value
                best_holder["record"] = record
            return value

        search = refine_scales(
            objective,
            init_scales,
            iters=gate["refine_iters"],
            step_size=TELEPORT_REFINE_STEP,
            spread_limit=gate["spread"],
            generator=generator,
        )
        best_refined = None
        if best_holder["record"] is not None:
            best_refined = dict(best_holder["record"])
            best_refined["n_accepted"] = search["n_accepted"]
            best_refined["n_infeasible"] = search["n_infeasible"]

        model.zero_grad(set_to_none=True)
        restore_training_state(model, optimizers, snap)
        model.train()

        samples = [r for _s, r in draws]
        return {
            "step": step_index,
            "base_loss": base_loss,
            "base_total_grad": base["total"],
            "base_nuclear_sum": base["nuclear_sum"],
            "base_fro": base["fro"],
            "base_nuclear": base["nuclear"],
            "samples": samples,
            "n_feasible": len(feasible),
            "max_abs_rel_dloss": max(abs(r["rel_dloss"]) for r in samples),
            "best_random": best_random,
            "best_refined": best_refined,
        }

    import time as _time

    snapshots: Dict[str, Dict[str, Any]] = {}
    probe_seconds = 0.0  # measurement overhead, kept out of time_seconds
    pending: list = []  # cached probe batches the base must still consume
    remaining = list(gate["snapshot_steps"])

    step = 0
    start_timer()
    while True:
        if remaining and step == remaining[0]:
            t_s = remaining.pop(0)
            stop_timer()
            cache = [
                pending.pop(0) if pending else next(stream)
                for _ in range(gate["probe_batches"])
            ]
            t_probe = _time.perf_counter()
            snapshots[str(t_s)] = teleport_probe(t_s, cache)
            torch.cuda.synchronize()
            probe_seconds += _time.perf_counter() - t_probe
            pending = cache + pending  # base resumes through the same batches
            start_timer()
        if step >= total_train_steps:
            break
        inputs, labels = pending.pop(0) if pending else next(stream)
        train_step(inputs, labels, step)
        step += 1
    stop_timer()

    final_val_acc = ab.evaluate(model, test_loader, tta_level=0)
    final_tta_val_acc = (
        ab.evaluate(model, test_loader, tta_level=tta_level) if tta_level else None
    )

    return {
        "optimizer": "vendor_muon",
        "epochs": epochs,
        "steps": step,
        "lr_schedule": "linear",
        "gate": dict(gate),
        "invariance_tol": TELEPORT_INVARIANCE_TOL,
        "refine_step_size": TELEPORT_REFINE_STEP,
        "gate_seed": gate_seed,
        "pair_names": pair_names(pairs),
        "snapshots": snapshots,
        "final_val_acc": final_val_acc,
        "final_tta_val_acc": final_tta_val_acc,
        "time_seconds": time_seconds,  # training only
        "probe_seconds": probe_seconds,  # orbit measurement overhead
    }


CF_KEYS = {"lr_scale", "enabled", "refresh_every", "k_directions", "beta_scale"}
# Probe-slice size for the central-flow curvature refresh: the third-order
# chain's memory scales with the forward graph, and curvature estimates do
# not need the full 2000-sample batch. Constant, not a config key (the cf:
# block is pinned to exactly five keys).
CF_PROBE_SAMPLES = 256


def _resolve_cf(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the ``cf:`` block of a central-flow config (CPU-safe)."""
    cf = config.get("cf")
    if not isinstance(cf, dict) or set(cf) != CF_KEYS:
        raise SystemExit(
            "experiment 'airbench_centralflow' needs a cf: block with exactly "
            f"the keys {sorted(CF_KEYS)}"
        )
    out = {
        "lr_scale": float(cf["lr_scale"]),
        "enabled": bool(cf["enabled"]),
        "refresh_every": int(cf["refresh_every"]),
        "k_directions": int(cf["k_directions"]),
        "beta_scale": float(cf["beta_scale"]),
    }
    if not 0.0 < out["lr_scale"] <= 1.0:
        raise SystemExit(f"cf.lr_scale must be in (0, 1], got {out['lr_scale']}")
    if out["refresh_every"] < 1:
        raise SystemExit("cf.refresh_every must be >= 1")
    if not 1 <= out["k_directions"] <= 16:
        raise SystemExit("cf.k_directions must be in [1, 16]")
    if out["beta_scale"] < 0.0:
        raise SystemExit("cf.beta_scale must be >= 0")
    return out


def momentum_topk_directions(momentum: torch.Tensor, k: int):
    """Top-k singular directions of a (2-D-reshaped) momentum buffer as
    rank-1 matrices ``u_i v_i^T`` (unit Frobenius norm each).

    These are the directions Muon's polar update weights equally and where
    the per-direction iterate oscillation of amplitude ~eta lives (finding
    F3/T3, docs/litreview/j-theory-theorem-sweep.md); the central-flow v0
    penalizes directional curvature exactly there.
    """
    # SVD on CPU: cuSOLVER's gesvd fails to converge on the ill-conditioned
    # early-training momentum buffers (observed on the first GPU smoke);
    # LAPACK is robust and these matrices are small. sharpgrad moves the
    # directions to the parameter device itself.
    m = momentum.detach().float().reshape(momentum.shape[0], -1).cpu()
    if not torch.isfinite(m).all():
        return []  # fp16 overflow in an early buffer: skip this refresh
    k = min(k, min(m.shape))
    try:
        u, _s, vh = torch.linalg.svd(m, full_matrices=False)
    except torch.linalg.LinAlgError:
        return []  # degenerate buffer: skip this matrix for this refresh
    return [
        torch.outer(u[:, i], vh[i, :]).reshape(momentum.shape) for i in range(k)
    ]


def run_airbench_centralflow(
    config: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Central-flow Muon v0 — flow-first program item 2 (litreview j §6).

    The mechanism test: does an EXPLICIT central-flow curvature-penalty term
    (src/optim/centralflow.py) at reduced LR reproduce what high-LR
    oscillating training achieves implicitly? Stock airbench recipe with the
    Muon (and SGD) LRs scaled by ``cf.lr_scale``, stock linear schedule
    shape; when ``cf.enabled``, every ``cf.refresh_every`` steps the penalty
    gradient ``grad_w sum_i w_i * v_i^T H v_i`` is recomputed on the current
    batch over the top-``cf.k_directions`` momentum singular directions of
    each filter matrix, with the theory-grounded v0 weights

        w_i = eta_t^2 / 2        (eta_t = current scaled Muon LR)

    — Muon's per-direction oscillation amplitude is ~eta (our F3/T3 bounded-
    update finding), so eta^2 is the iterate-oscillation variance the central
    flow says the oscillation would contribute. The cached penalty is applied
    every step with beta = eta_t * cf.beta_scale.

    ``recipe.compile`` defaults OFF here: the refresh runs a third-order
    autograd chain through the model forward, which torch.compile does not
    reliably support; arms are compared uncompiled-vs-uncompiled.

    Arms are config files varying (lr_scale, enabled); dev-seed measurement
    experiment; never a comparison-table entry.
    """
    from src.optim.centralflow import CentralFlowTerm

    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench_centralflow' pins the stock recipe; it "
            "does not accept an optimizer override"
        )
    cf = _resolve_cf(config)
    sampling = _resolve_sampling(config.get("recipe", {}))
    if sampling is not None:
        raise SystemExit("airbench_centralflow supports vendored sampling only")

    if device.type != "cuda":
        raise SystemExit("airbench_centralflow requires a CUDA device")

    ab = load_vendor_airbench()

    train_cfg = config.get("train", {})
    recipe_cfg = config.get("recipe", {})
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))
    epochs = float(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 2000))
    bias_lr = float(recipe_cfg.get("bias_lr", 0.053))
    head_lr = float(recipe_cfg.get("head_lr", 0.67))
    wd = float(recipe_cfg.get("sgd_weight_decay", 2e-6)) * batch_size
    tta_level = int(recipe_cfg.get("tta_level", 2))

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if bool(recipe_cfg.get("compile", False)):  # default OFF (third-order chain)
        model.compile()

    test_loader = ab.CifarLoader(data_root, train=False, batch_size=2000)
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    total_train_steps = math.ceil(epochs * len(train_loader))
    whiten_bias_train_steps = min(math.ceil(3 * len(train_loader)), total_train_steps)

    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    # LR scale applies to every scheduled group (Muon AND the SGD groups):
    # the arm is "the same recipe, colder", not a per-group reweighting.
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    optimizer2 = ab.Muon(
        filter_params, lr=AIRBENCH_STOCK_LR, momentum=0.6, nesterov=True
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"] * cf["lr_scale"]

    term = CentralFlowTerm() if cf["enabled"] else None

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    cf_timeseries = []
    val_accs = []
    train_acc = float("nan")
    for _epoch in range(math.ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            torch.nn.functional.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="sum"
            ).backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (
                    1 - step / whiten_bias_train_steps
                ) / cf["lr_scale"]  # whiten bias keeps its stock, unscaled decay
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            muon_lr = optimizer2.param_groups[0]["lr"]
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            if term is not None:
                if step % cf["refresh_every"] == 0:
                    # Memory-bounded refresh: one chunk per filter matrix,
                    # each with its own forward on a small probe slice of the
                    # current batch — a joint refresh over all matrices on the
                    # full 2000-sample graph OOMs a 48 GB card (third-order
                    # chains hold the full graph per direction set).
                    b_inputs = inputs[:CF_PROBE_SAMPLES]
                    b_labels = labels[:CF_PROBE_SAMPLES]

                    def make_loss_fn():
                        def loss_fn():
                            model.train()
                            out = model(
                                b_inputs,
                                whiten_bias_grad=False,  # frozen past epoch 3;
                                # CF only ever touches filter params anyway
                            )
                            return torch.nn.functional.cross_entropy(
                                out.float(),
                                b_labels,
                                label_smoothing=0.2,
                                reduction="sum",
                            )

                        return loss_fn

                    chunks = []
                    for p in filter_params:
                        buf = optimizer2.state.get(p, {}).get("momentum_buffer")
                        if buf is None:
                            continue
                        dirs = [
                            [
                                d.to(q.dtype) if q is p else None
                                for q in filter_params
                            ]
                            for d in momentum_topk_directions(
                                buf, cf["k_directions"]
                            )
                        ]
                        if not dirs:
                            continue  # degenerate/non-finite buffer this step
                        weights = [0.5 * muon_lr**2] * len(dirs)
                        chunks.append(
                            (make_loss_fn(), filter_params, dirs, weights)
                        )
                    if chunks:
                        term.refresh_from_chunks(chunks, step=step)
                if term.penalty_grads is not None:
                    term.apply(filter_params, beta=muon_lr * cf["beta_scale"])
                    if step % ROUTING_TS_EVERY == 0:
                        stats = term.stats()
                        cf_timeseries.append(
                            {
                                "step": step,
                                "muon_lr": muon_lr,
                                "n_directions": stats["n_directions"],
                                "penalty_grad_norm": stats["penalty_grad_norm"],
                                "curvature_mean": (
                                    sum(stats["curvatures"])
                                    / max(1, len(stats["curvatures"]))
                                ),
                                "curvature_max": (
                                    max(stats["curvatures"])
                                    if stats["curvatures"]
                                    else 0.0
                                ),
                            }
                        )
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_accs.append(ab.evaluate(model, test_loader, tta_level=0))
        if step >= total_train_steps:
            break

    start_timer()
    tta_val_acc = (
        ab.evaluate(model, test_loader, tta_level=tta_level) if tta_level else None
    )
    stop_timer()

    return {
        "optimizer": "vendor_muon",
        "epochs": epochs,
        "steps": step,
        "cf": dict(cf),
        "weight_mode": "eta_sq_half_topk_momentum",
        "train_acc_last": train_acc,
        "val_accs": val_accs,
        "val_acc": val_accs[-1],
        "tta_val_acc": tta_val_acc,
        "cf_timeseries": cf_timeseries,
        "time_seconds": time_seconds,
    }


AIRBENCH_STOCK_LR = 0.24  # vendored record hyperparameter (airbench94_muon.py:362)

# Gate-1 amendment A4 (mechanism probes): the ONLY optimizer keys a
# 'probe_overrides' block may touch in the instrumented experiment.
PROBE_OVERRIDE_KEYS = ("lr", "momentum", "nesterov")


def _validate_probe_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the A4 ``probe_overrides:`` block of an instrumented config.

    Returns the (possibly empty) override dict. CPU-safe (called before any
    CUDA work). Refuses: non-dict/empty blocks, keys outside
    ``PROBE_OVERRIDE_KEYS``, an 'eval' sweep-seed policy, and a base config
    seed < 1000 -- mechanism probes are dev-seed measurement runs, never
    comparison-table entries.
    """
    probe = config.get("probe_overrides")
    if probe is None:
        return {}
    if not isinstance(probe, dict) or not probe:
        raise SystemExit(
            "probe_overrides must be a non-empty mapping of vendored-Muon "
            f"hyperparameters (allowed keys: {PROBE_OVERRIDE_KEYS})"
        )
    unknown = sorted(set(probe) - set(PROBE_OVERRIDE_KEYS))
    if unknown:
        raise SystemExit(
            f"probe_overrides may only touch {PROBE_OVERRIDE_KEYS}; "
            f"got unknown key(s): {unknown}"
        )
    sweep_spec = (config.get("sweep") or {}).get("seeds")
    policy = (
        sweep_spec
        if isinstance(sweep_spec, str)
        else sweep_spec.get("policy")
        if isinstance(sweep_spec, dict)
        else None
    )
    if policy == "eval":
        raise SystemExit(
            "probe_overrides configs are dev-seed mechanism probes (Gate-1 "
            "amendment A4) and must never use the eval seed policy"
        )
    seed = config.get("seed")
    if isinstance(seed, int) and seed < 1000:
        raise SystemExit(
            f"probe_overrides config carries seed {seed} < 1000; mechanism "
            "probes are dev-seed only"
        )
    return dict(probe)


def run_airbench_instrumented(
    config: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """WP1.2 instrumented airbench: the IDENTICAL stock WP0.1 recipe
    (vendored Muon lr=0.24 momentum=0.6 nesterov, compile on) with an
    InstrumentationHub observing the filter-parameter matrices.

    Per plan section 1.1: top-k1 + k2 bulk tracked pairs of the vendored
    Muon's momentum buffers; per-step raw PRE-momentum gradient projections
    s_i = u^T G v (grabbed via ``hub.capture_grads()`` before
    ``optimizer2.step()``); per-matrix top sigma and ||G||_F; both betas.

    HVP probes (``instrumentation.hvp: true``, default false): once per
    tracked pair per refresh, lambda_i = vec(u_i v_i^T)^T H vec(u_i v_i^T)
    restricted to that matrix, on the CURRENT batch, via
    :class:`src.instrument.hvp.AirbenchHvpProbe` (fp32 functional re-forward
    + double-backward; read-only w.r.t. training).  Phase-1 VALIDATION ONLY
    -- they calibrate the amplitude-ratio implied-eta*lambda estimator and
    are forbidden in any optimizer update path.  Requires
    ``recipe.compile: false`` (double-backward through torch.compile is
    unsupported; the config must opt out of the stock compile explicitly).

    Directional-smoothness probe (``instrumentation.smoothness``, default
    off): every ``t_meas`` steps, the SPECTRAL-norm and Euclidean directional
    smoothness of the minibatch loss along the actual applied update, per
    Muon-managed matrix (:class:`src.instrument.smoothness.SmoothnessProbe`;
    pre-registered question in that module's docstring).  Like ``hvp`` it
    requires ``recipe.compile: false``, and it is intended to run TOGETHER
    with ``hvp: true`` so one run set yields both the Euclidean eta*lambda and
    the generalized (spectral) smoothness.

    Frozen-probe tier (``instrumentation.frozen_probes``, default off): k3
    never-refreshed random probe directions per matrix with unbounded-window
    cumulative t-statistics (:class:`src.instrument.tracker.FrozenProbeBank`).

    Zero behavior change to training itself: the hub is read-only and the
    recipe, schedules, and optimizers are exactly those of the ``airbench``
    experiment.  The returned metrics carry the full instrumentation log
    under the private key ``"_instrumentation_log"``; scripts/run.py pops it
    and writes the src.instrument.schema sidecar next to the results JSON.

    MECHANISM PROBES (Gate-1 amendment A4): a clearly-marked top-level
    ``probe_overrides:`` block may override ONLY the vendored-Muon
    hyperparameters lr / momentum / nesterov, for the instrumented mechanism
    probes (momentum=0 run, LR ladder). This is the single sanctioned
    deviation from the hard-pinned stock record recipe, honored by THIS
    experiment only (``run_airbench_smoke``/``run_airbench`` never read it).
    Dev-seeds only: a config carrying ``probe_overrides`` is refused if its
    ``sweep.seeds`` policy is 'eval' or its base ``seed`` is < 1000
    (materialized sweep variants inherit dev seeds from the source config,
    whose expansion already enforces this via scripts/sweep.py). Probe runs
    are measurement, never comparison-table entries. The effective optimizer
    hyperparameters are recorded in metrics (``optimizer_lr``,
    ``probe_overrides``) so downstream eta*lambda analyses use the real lr.

    LAUNCH PRECONDITION (WP1.2, enforced in scripts/run.py): the
    human-authored ``criteria/phase1_preregistration.md`` must exist before
    any run of this experiment.
    """
    from src.instrument import hub_from_config

    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench_instrumented' is the stock WP0.1 recipe "
            "plus read-only instrumentation; it does not accept an optimizer "
            "override. (Mechanism probes use the restricted 'probe_overrides' "
            "block instead.)"
        )
    probe = _validate_probe_overrides(config)
    instr_cfg = config.get("instrumentation")
    if not isinstance(instr_cfg, dict):
        raise SystemExit(
            "experiment 'airbench_instrumented' requires an 'instrumentation' "
            "block in the config (k1, k2, t_refresh, betas, classifier, ...)"
        )

    merged = dict(config)
    merged.pop("probe_overrides", None)  # consumed here, never forwarded
    merged["optimizer"] = dict(
        name="vendor_muon", lr=AIRBENCH_STOCK_LR, momentum=0.6, nesterov=True
    )
    if probe:
        merged["optimizer"].update(probe)  # A4 mechanism probe, dev-only
    recipe = dict(config.get("recipe", {}))
    recipe["normalize_filter_weights"] = False  # vendored Muon.step does it
    recipe.setdefault("compile", True)  # the reference compiles the model
    merged["recipe"] = recipe

    hvp_requested = bool(instr_cfg.get("hvp", False))
    if hvp_requested and recipe.get("compile", True):
        raise SystemExit(
            "instrumentation.hvp: true requires recipe.compile: false -- "
            "double-backward (create_graph) through a torch.compile'd model "
            "is unsupported; the HVP calibration run must opt out of the "
            "stock compile explicitly in its config."
        )
    smoothness_cfg = instr_cfg.get("smoothness")
    smoothness_requested = bool(
        smoothness_cfg
        if isinstance(smoothness_cfg, bool)
        else (smoothness_cfg or {}).get("enabled", bool(smoothness_cfg))
    )
    if smoothness_requested and recipe.get("compile", True):
        raise SystemExit(
            "instrumentation.smoothness requires recipe.compile: false -- the "
            "probe re-evaluates the loss through torch.func.functional_call "
            "(and, with grad_source: recompute, differentiates it), which is "
            "not supported through a torch.compile'd model."
        )

    holder: Dict[str, Any] = {}

    def factory(model, optimizer2, filter_params):
        names = {id(p): n for n, p in model.named_parameters()}
        named = [
            (names.get(id(p), f"filter_{i}"), p)
            for i, p in enumerate(filter_params)
        ]
        hvp_fn = None
        if hvp_requested:
            # Phase-1 validation only; lives in src.instrument (never
            # importable from any optimizer update path).
            from src.instrument.hvp import AirbenchHvpProbe

            hvp_fn = holder["hvp_probe"] = AirbenchHvpProbe(
                model, filter_params, label_smoothing=0.2
            )
        if smoothness_requested:
            # Trajectory directional smoothness in the spectral norm (the
            # quantity the non-Euclidean EoS theory says governs Muon) plus
            # its Euclidean twin, measured on the SAME runs as the HVP
            # eta*lambda -- that side-by-side is the whole point.
            from src.instrument.smoothness import smoothness_from_config

            holder["smoothness"] = smoothness_from_config(
                instr_cfg, model, named, label_smoothing=0.2
            )
        holder["hub"] = hub_from_config(instr_cfg, named, optimizer2, hvp_fn=hvp_fn)
        return holder["hub"]

    batch_hook = None
    if hvp_requested or smoothness_requested:

        def batch_hook(inputs, labels):
            if hvp_requested:
                holder["hvp_probe"].set_batch(inputs, labels)
            probe = holder.get("smoothness")
            if probe is not None:
                probe.set_batch(inputs, labels)

    pre_hook = post_hook = None
    if smoothness_requested:

        def pre_hook(step):
            probe = holder.get("smoothness")
            if probe is not None:
                probe.before_step(step)

        def post_hook(step, lr):
            probe = holder.get("smoothness")
            if probe is not None:
                probe.after_step(step, lr)

    metrics = run_airbench_smoke(
        merged,
        device,
        _hub_factory=factory,
        _batch_hook=batch_hook,
        _pre_step_hook=pre_hook,
        _post_step_hook=post_hook,
    )
    metrics["instrumented"] = True
    if hvp_requested:
        metrics["hvp_graph_builds"] = holder["hvp_probe"].n_graph_builds
    # Effective lr for the eta*lambda plot (equals the stock record lr unless
    # an A4 probe override changed it -- analyses must use the real value).
    metrics["optimizer_lr"] = merged["optimizer"]["lr"]
    if probe:
        metrics["probe_overrides"] = dict(probe)
    log = holder["hub"].to_log()
    probe = holder.get("smoothness")
    if probe is not None:
        log["smoothness"] = probe.to_log()
        metrics["smoothness_forward_passes"] = probe.n_forward
        metrics["smoothness_backward_passes"] = probe.n_backward
    metrics["_instrumentation_log"] = log
    return metrics


# ----------------------------------------------------------------- entrypoint


def _load_run_module():
    """Import scripts/run.py (the WP0.0 runner) without modifying it."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "run.py"
    spec = importlib.util.spec_from_file_location("routed_muon_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    """Delegate to scripts/run.py with the zoo registered at runtime.

    Usage: uv run python -m src.optim.airbench_zoo <config.yaml> [--seed N]
    """
    from src.optim.registry import OPTIMIZER_REGISTRY

    run_mod = _load_run_module()
    run_mod.OPTIMIZER_REGISTRY.update(OPTIMIZER_REGISTRY)
    run_mod.EXPERIMENT_REGISTRY["airbench_smoke"] = run_airbench_smoke
    run_mod.EXPERIMENT_REGISTRY["airbench"] = run_airbench
    run_mod.EXPERIMENT_REGISTRY["airbench_instrumented"] = run_airbench_instrumented
    return run_mod.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
