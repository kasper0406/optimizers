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
    sampling = _resolve_sampling(config.get("recipe", {}))

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
    # Program #10 (branched-probe meta-control) Phase A: mid-run LR fork.
    # Two runs with the same seed are bitwise-identical until ``step``; a
    # multiplier applied from ``step`` onward makes them a perfectly
    # common-randomness-paired branch pair at full-run cost (~15 s here).
    lr_fork = recipe_cfg.get("lr_fork") or None  # {"step": int, "mult": float}
    if lr_fork is not None:
        lr_fork = {"step": int(lr_fork["step"]), "mult": float(lr_fork["mult"])}
    # Program #12 Phase A (reports/data-selection-prereg.md): per-example
    # momentum-alignment probe. Measurement-only; eval-mode forwards (BN
    # running stats untouched); grads computed in a separate autograd pass
    # between optimizer steps. {"n_fixed": int, "n_fresh": int, "every": int,
    # "topk": int}.
    example_probe = recipe_cfg.get("example_probe") or None

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
    train_loss_series: list = []  # P10: populated only when lr_fork is set
    probe_rows: list = []  # P12: per-example momentum-alignment records

    def _run_example_probe(step_now: int, batch_inputs, batch_labels) -> None:
        n_fixed = int(example_probe.get("n_fixed", 64))
        n_fresh = int(example_probe.get("n_fresh", 64))
        topk = int(example_probe.get("topk", 8))
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        # momentum buffers + their top-k singular pairs, fixed for this probe
        bufs, bases = {}, {}
        for p in filter_params:
            st = optimizer2.state.get(p, {})
            m = st.get("momentum_buffer")
            if m is None:
                continue
            m2 = m.reshape(len(m), -1).float()
            bufs[p] = m2
            u, s, v = torch.svd_lowrank(m2, q=min(topk, min(m2.shape)))
            bases[p] = (u, v)
        fixed_x = train_loader.images[:n_fixed]
        fixed_y = train_loader.labels[:n_fixed]
        sets = [("fixed", fixed_x, fixed_y),
                ("fresh", batch_inputs[:n_fresh], batch_labels[:n_fresh])]
        for which, xs, ys in sets:
            for i in range(len(xs)):
                model.zero_grad(set_to_none=True)
                out = model(xs[i:i + 1])
                li = torch.nn.functional.cross_entropy(
                    out.float(), ys[i:i + 1], label_smoothing=0.2
                )
                li.backward()
                row = {"step": step_now, "which": which, "idx": i,
                       "loss": float(li.detach()), "m": []}
                for p in filter_params:
                    if p.grad is None or p not in bufs:
                        continue
                    g2 = p.grad.reshape(len(p.grad), -1).float()
                    m2 = bufs[p]
                    gn = g2.norm().clamp_min(1e-30)
                    cos = float((g2 * m2).sum() / (gn * m2.norm().clamp_min(1e-30)))
                    u, v = bases[p]
                    coef = (u.T @ g2 @ v).diagonal()
                    frac = float((coef ** 2).sum() / gn ** 2)
                    row["m"].append({"cos": round(cos, 5), "topk_frac": round(frac, 5)})
                probe_rows.append(row)
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

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

    val_accs = []
    train_acc = float("nan")
    for _epoch in range(math.ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss = torch.nn.functional.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="sum"
            )
            loss.backward()
            if lr_fork is not None:
                # P10: per-step loss series (GPU-resident; synced once at end).
                # Recomputed in fp32: the recipe's fp16 sum-reduction loss
                # overflows to inf mid-run (harmless to training — CE backward
                # never uses the forward value — but useless as telemetry).
                with torch.no_grad():
                    train_loss_series.append(
                        torch.nn.functional.cross_entropy(
                            outputs.detach().float(), labels,
                            label_smoothing=0.2, reduction="sum",
                        )
                    )
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (
                    1 - step / whiten_bias_train_steps
                )
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            if lr_fork is not None and step >= lr_fork["step"]:
                # P10: scale ALL optimizer LRs from the fork step on (the
                # branch differs from its same-seed twin only in this).
                for group in optimizer1.param_groups + optimizer2.param_groups:
                    group["lr"] = group["lr"] * lr_fork["mult"]
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
            model.zero_grad(set_to_none=True)
            step += 1
            if example_probe is not None and step % int(example_probe.get("every", 10)) == 0:
                _run_example_probe(step, inputs, labels)
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
        if step >= total_train_steps:
            break

    start_timer()
    tta_val_acc = (
        ab.evaluate(model, test_loader, tta_level=tta_level) if tta_level else None
    )
    stop_timer()

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
    if track_routing:
        # Gate-1 amendment A5: end-of-run routing telemetry + coarse series.
        metrics["routing_stats"] = optimizer2.routing_stats()
        metrics["routing_timeseries"] = routing_timeseries
    if hasattr(optimizer2, "tempo_stats"):
        # Program #8: TempoMuon rho/gain telemetry (self-recorded per step).
        metrics["tempo_stats"] = optimizer2.tempo_stats()
    if lr_fork is not None:
        metrics["lr_fork"] = lr_fork
        metrics["train_loss_series"] = [
            float(v) for v in torch.stack(train_loss_series).cpu()
        ]
    if example_probe is not None:
        metrics["example_probe"] = {"config": example_probe, "rows": probe_rows}
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
