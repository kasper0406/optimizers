"""With-replacement sampling ablation tests (Phase-1 disambiguation, Task 1).

CPU-only: the vendored CifarLoader itself needs CUDA, so the iteration logic
is exercised on a stub loader carrying the exact attributes the vendored
``CifarLoader.__iter__`` uses -- which also lets the vendored iterator run on
the SAME stub (``CifarLoader.__iter__(stub)``) for an exact augmentation-
parity check against :func:`iter_batches_with_replacement`.

Default-path bit-identity: the with-replacement code is only engaged behind
``recipe.sampling: with_replacement`` (``_resolve_sampling`` returns None for
an absent key and the wrapper is constructed only inside that branch); the
vendored module is untouched and every pre-existing airbench test runs
unchanged.

Seeds: dev seeds only (>= 1000).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util

import pytest
import torch
import yaml

from src.optim import airbench_zoo
from src.optim.airbench_zoo import (
    WithReplacementLoader,
    _resolve_sampling,
    iter_batches_with_replacement,
    run_airbench_smoke,
)

ab = airbench_zoo.load_vendor_airbench()

WR_CONFIG = REPO_ROOT / "configs" / "dev" / "instrumented_airbench_withreplacement.yaml"


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("sweep_module_sampling", "scripts/sweep.py")


class StubLoader:
    """CPU stand-in exposing exactly the attributes CifarLoader.__iter__ and
    iter_batches_with_replacement consume."""

    def __init__(self, n=64, batch_size=16, aug=None, side=8, seed=1400):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, side, side, generator=gen)
        self.labels = torch.arange(n)
        self.normalize = lambda x: x  # identity; the caching path still runs
        self.proc_images = {}
        self.epoch = 0
        self.aug = dict(aug or {})
        self.batch_size = batch_size
        self.drop_last = True
        self.shuffle = True

    def __len__(self):
        return len(self.images) // self.batch_size


# ------------------------------------------------------------ flag resolution


def test_resolve_sampling_default_is_vendor_behavior():
    assert _resolve_sampling({}) is None
    assert _resolve_sampling({"sampling": None}) is None


def test_resolve_sampling_accepts_with_replacement():
    assert _resolve_sampling({"sampling": "with_replacement"}) == "with_replacement"


def test_resolve_sampling_rejects_unknown_values():
    with pytest.raises(SystemExit, match="recipe.sampling"):
        _resolve_sampling({"sampling": "without_replacement"})


def test_run_airbench_smoke_validates_sampling_before_device():
    """Invalid sampling is a config error, reported even on the CPU dev box
    (before the CUDA guard)."""
    cfg = {"recipe": {"sampling": "bogus"}}
    with pytest.raises(SystemExit, match="recipe.sampling"):
        run_airbench_smoke(cfg, torch.device("cpu"))


def test_run_airbench_smoke_valid_sampling_reaches_cuda_guard():
    """With the flag valid (or absent) the harness proceeds to the usual
    CUDA requirement -- the sampling code changed nothing before it."""
    for recipe in ({}, {"sampling": "with_replacement"}):
        with pytest.raises(SystemExit, match="CUDA"):
            run_airbench_smoke({"recipe": recipe}, torch.device("cpu"))


# --------------------------------------------------------- index distribution


def test_with_replacement_indices_in_range_and_images_match():
    """No augmentation: yielded rows must be exact rows of the source set,
    labels acting as the drawn indices (labels == arange)."""
    stub = StubLoader(n=64, batch_size=16, aug=None)
    torch.manual_seed(1401)
    all_labels = []
    for images, labels in WithReplacementLoader(stub, ab):
        assert images.shape == (16, 3, 8, 8)
        assert labels.min() >= 0 and labels.max() < 64
        assert torch.equal(images, stub.images[labels])
        all_labels.append(labels)
    drawn = torch.cat(all_labels)
    assert len(drawn) == 64  # 4 batches of 16: steps per epoch unchanged


def test_with_replacement_produces_duplicates_within_an_epoch():
    """64 i.i.d. draws from 64 items are all-distinct with probability
    64!/64^64 ~ 1e-27; a without-replacement epoch is always all-distinct."""
    stub = StubLoader(n=64, batch_size=16, aug=None)
    torch.manual_seed(1402)
    drawn = torch.cat([lab for _, lab in WithReplacementLoader(stub, ab)])
    assert len(torch.unique(drawn)) < len(drawn)


def test_with_replacement_draws_are_roughly_uniform():
    stub = StubLoader(n=16, batch_size=16, aug=None)
    torch.manual_seed(1403)
    wr = WithReplacementLoader(stub, ab)
    counts = torch.zeros(16, dtype=torch.long)
    n_epochs = 400
    for _ in range(n_epochs):
        for _, labels in wr:
            counts += torch.bincount(labels, minlength=16)
    total = counts.sum().item()
    assert total == n_epochs * 16
    freqs = counts.float() / total
    # Expected 1/16 = 0.0625; 6400 draws => se ~ 0.003 per bin. Loose bounds.
    assert freqs.min() > 0.045
    assert freqs.max() < 0.080


# ------------------------------------------------------- augmentation parity


def _index_to_image_map(batches):
    """labels are arange indices, so each yielded row identifies the
    augmented image of its underlying index for that epoch."""
    mapping = {}
    for images, labels in batches:
        for row, lab in zip(images, labels.tolist()):
            mapping[lab] = row
    return mapping


@pytest.mark.parametrize("aug", [dict(flip=True, translate=2), dict(flip=True), {}])
def test_augmentation_treatment_identical_to_vendor_per_epoch(aug):
    """Per epoch, the with-replacement iterator must serve each index the
    EXACT augmented image the vendored iterator would serve it: identical
    epoch-0 preprocessing cache, identical per-epoch crop of the whole set,
    identical every-other-epoch deterministic flip. Both iterators run on
    identically-initialized stubs with the SAME RNG seed per epoch, so the
    augmentation draws coincide and only the index draw differs."""
    stub_vendor = StubLoader(n=32, batch_size=8, aug=aug)
    stub_wr = StubLoader(n=32, batch_size=8, aug=aug)
    assert torch.equal(stub_vendor.images, stub_wr.images)

    for epoch, seed in enumerate([1404, 1405, 1406]):
        torch.manual_seed(seed)
        vendor_map = _index_to_image_map(ab.CifarLoader.__iter__(stub_vendor))
        torch.manual_seed(seed)
        wr_batches = list(iter_batches_with_replacement(stub_wr, ab))

        assert stub_vendor.epoch == stub_wr.epoch == epoch + 1
        assert len(wr_batches) == len(stub_wr)
        for images, labels in wr_batches:
            for row, lab in zip(images, labels.tolist()):
                assert torch.equal(row, vendor_map[lab]), (
                    f"epoch {epoch}: augmented image for index {lab} differs "
                    "from the vendored augmentation path"
                )


def test_wrapper_len_matches_vendor_len():
    stub = StubLoader(n=64, batch_size=16)
    assert len(WithReplacementLoader(stub, ab)) == len(stub) == 4


# ----------------------------------------------------------------- the config


def test_withreplacement_config_parses_and_expands_to_seeds_1100_1102():
    with open(WR_CONFIG) as fh:
        config = yaml.safe_load(fh)
    assert config["experiment"] == "airbench_instrumented"
    assert config["recipe"]["sampling"] == "with_replacement"
    # Identical instrumentation settings to the WP1.2 baseline config.
    with open(REPO_ROOT / "configs" / "wp12_airbench_instrumented.yaml") as fh:
        wp12 = yaml.safe_load(fh)
    assert config["instrumentation"] == wp12["instrumentation"]
    assert config["train"] == wp12["train"]

    sweep.refuse_eval_seed_literals(config, str(WR_CONFIG))  # must not raise
    plan = sweep.expand_sweep(config, WR_CONFIG)
    assert plan["seed_policy"] == "explicit-dev"
    assert plan["seeds"] == [1100, 1101, 1102]
    assert len(plan["runs"]) == 3
    assert "seed" not in plan["variants"][0]["config"]
    assert (
        plan["variants"][0]["config"]["recipe"]["sampling"] == "with_replacement"
    )
