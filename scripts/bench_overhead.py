#!/usr/bin/env python
"""WP1.1 overhead benchmark: instrumented vs stock step time.

Two modes:

* ``synthetic`` (default; any device, incl. CPU): a stack of weight matrices
  shaped like the airbench94 filter matrices, driven by a real forward/backward
  through a small model built from those matrices, stepped by the zoo Muon.
  Runs everywhere; used for correctness and for local dev timing.
* ``airbench`` (CUDA only): the vendored airbench94 net + CifarLoader, real
  data, real augmentation, our Muon on the filter params + SGD on
  biases/head -- the loop of vendor/airbench/airbench94_muon.py:main()
  without schedules/eval, timed with CUDA events. This is the mode the
  <10% overhead DoD is measured with, on a GPU box:

      uv run python scripts/bench_overhead.py --mode airbench --device cuda \\
          --config configs/dev/instrumented_airbench_muon.yaml \\
          --steps 200 --out results/bench_overhead_airbench.json

Each mode runs the same training loop twice -- stock, and with an
InstrumentationHub observing every step -- and reports median per-step wall
time plus the overhead fraction.  With ``--out`` a results-schema JSON
(src/results_io.py) is written (append-only; refuses overwrite).

Seed policy: the config carries a dev seed (>= 1000).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src import results_io
from src.instrument import InstrumentationHub, hub_from_config
from src.optim import Muon


# ----------------------------------------------------------------- utilities


def _timer(device: torch.device) -> Callable[[Callable[[], None]], float]:
    """Return a fn that times one call in milliseconds on the given device."""
    if device.type == "cuda":
        def cuda_time(fn: Callable[[], None]) -> float:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            return float(starter.elapsed_time(ender))
        return cuda_time

    def cpu_time(fn: Callable[[], None]) -> float:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1e3
    return cpu_time


def _summarize(stock_ms: List[float], instr_ms: List[float]) -> Dict[str, Any]:
    med_stock = statistics.median(stock_ms)
    med_instr = statistics.median(instr_ms)
    return {
        "stock_step_ms_median": med_stock,
        "instrumented_step_ms_median": med_instr,
        "stock_step_ms_mean": statistics.fmean(stock_ms),
        "instrumented_step_ms_mean": statistics.fmean(instr_ms),
        "overhead_ms_median": med_instr - med_stock,
        "overhead_frac_median": (med_instr - med_stock) / med_stock,
        "n_timed_steps": len(stock_ms),
    }


# ------------------------------------------------------------ synthetic mode


class _MatrixStackModel(torch.nn.Module):
    """Minimal model exercising a list of weight matrices with a real
    forward/backward: y = W_i x_i per matrix, summed scalar loss."""

    def __init__(self, shapes: List[Tuple[int, int]]):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(m, n) / (n**0.5)) for m, n in shapes]
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        out = 0.0
        for W, x in zip(self.weights, xs):
            out = out + torch.tanh(W @ x).square().mean()
        return out


def _run_synthetic(
    config: Dict[str, Any],
    device: torch.device,
    steps: int,
    warmup: int,
    instrumented: bool,
) -> List[float]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    shapes = [tuple(s) for s in config["model"]["matrix_shapes"]]
    batch = int(config["train"].get("batch_size", 64))
    opt_cfg = dict(config["optimizer"])
    opt_cfg.pop("name", None)
    ns_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = _MatrixStackModel(shapes).to(device)
    optimizer = Muon(model.parameters(), ns_dtype=ns_dtype, **opt_cfg)
    hub: Optional[InstrumentationHub] = None
    if instrumented:
        hub = hub_from_config(
            config["instrumentation"], list(model.named_parameters()), optimizer
        )

    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xs_all = [
        [torch.randn(n, batch, generator=gen).to(device) for _, n in shapes]
        for _ in range(4)  # cycle a small pool of fixed inputs
    ]

    timer = _timer(device)
    times: List[float] = []

    def one_step(i: int) -> None:
        xs = xs_all[i % len(xs_all)]
        loss = model(xs)
        loss.backward()
        optimizer.step()
        if hub is not None:
            hub.after_step()
        optimizer.zero_grad(set_to_none=True)

    for i in range(warmup + steps):
        ms = timer(lambda: one_step(i))
        if i >= warmup:
            times.append(ms)
    return times


# ------------------------------------------------------------- airbench mode


def _run_airbench(
    config: Dict[str, Any],
    device: torch.device,
    steps: int,
    warmup: int,
    instrumented: bool,
) -> List[float]:
    """Airbench94 training steps (vendored net + loader + recipe optimizers),
    timed with CUDA events. CUDA only (vendored loader maps data to cuda)."""
    if device.type != "cuda":
        raise SystemExit("--mode airbench requires --device cuda")
    from src.optim.airbench_zoo import load_vendor_airbench

    ab = load_vendor_airbench()
    torch.manual_seed(int(config["seed"]))

    batch_size = int(config.get("train", {}).get("batch_size", 2000))
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    model = ab.make_net()

    opt_cfg = dict(config["optimizer"])
    opt_cfg.pop("name", None)
    filter_params = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim != 4 and p.requires_grad]
    optimizer1 = torch.optim.SGD(other_params, lr=0.01, momentum=0.85, nesterov=True)
    optimizer2 = Muon(filter_params, **opt_cfg)

    hub: Optional[InstrumentationHub] = None
    if instrumented:
        named = [
            (f"filter_{i}", p) for i, p in enumerate(filter_params)
        ]
        hub = hub_from_config(config["instrumentation"], named, optimizer2)

    model.reset()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    model.train()

    timer = _timer(device)
    times: List[float] = []
    step = 0
    import torch.nn.functional as F

    while step < warmup + steps:
        for inputs, labels in train_loader:
            def one_step() -> None:
                outputs = model(inputs)
                F.cross_entropy(
                    outputs, labels, label_smoothing=0.2, reduction="sum"
                ).backward()
                optimizer1.step()
                optimizer2.step()
                if hub is not None:
                    hub.after_step()
                model.zero_grad(set_to_none=True)

            ms = timer(one_step)
            if step >= warmup:
                times.append(ms)
            step += 1
            if step >= warmup + steps:
                break
    return times


# ----------------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/dev/instrumented_mlp_smoke.yaml",
        help="config with optimizer + instrumentation blocks",
    )
    parser.add_argument("--mode", choices=("synthetic", "airbench"), default="synthetic")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--steps", type=int, default=50, help="timed steps per arm")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write a results-schema JSON here (append-only)",
    )
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    if int(config.get("seed", 0)) < 1000:
        raise SystemExit("config seed must be a dev seed (>= 1000)")
    device = torch.device(args.device)

    runner = _run_synthetic if args.mode == "synthetic" else _run_airbench
    started = results_io.utc_now_iso()
    t0 = time.perf_counter()
    stock_ms = runner(config, device, args.steps, args.warmup, instrumented=False)
    instr_ms = runner(config, device, args.steps, args.warmup, instrumented=True)
    wall = time.perf_counter() - t0

    metrics = _summarize(stock_ms, instr_ms)
    metrics["mode"] = args.mode
    metrics["note"] = (
        "overhead DoD (<10%) is evaluated on --mode airbench --device cuda; "
        "this file reports measurements only, pass/fail is a human judgment"
    )
    print(yaml.safe_dump(metrics, sort_keys=True))

    if args.out is not None:
        if device.type == "cuda":
            gpu = torch.cuda.get_device_name(device)
        else:
            gpu = device.type
        result = {
            "schema_version": results_io.SCHEMA_VERSION,
            "experiment": f"bench_overhead_{args.mode}",
            "config": results_io.config_record(args.config, config),
            **results_io.git_provenance(),
            "seed": int(config["seed"]),
            "gpu_type": gpu,
            "wall_time_s": wall,
            "cost_usd": None,
            "started_at": started,
            "finished_at": results_io.utc_now_iso(),
            "metrics": metrics,
        }
        results_io.write_result(result, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
