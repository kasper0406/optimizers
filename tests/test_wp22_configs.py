"""WP2.2/2.3 experiment-matrix config tests.

Pins the contract between configs/wp22_*.yaml + configs/wp23_lambda_tracking.yaml
and docs/wp22-run-plan.md (whose fenced-JSON manifest is the machine-readable
run plan):

- every config parses and its optimizer resolves in the registry;
- no literal seed < 1000 anywhere (CLAUDE.md rule 2; eval seeds 0-99 are
  additionally refused by scripts/sweep.py at expansion time);
- seed policies per group match the plan (eval for comparison tables, dev
  n=25 for selection stage A, dev n=10 for the LR stress probe);
- grid coverage / run counts match the run-plan manifest exactly;
- placeholder configs (stage B, retuned-WD null) and blocked configs
  (grad-clip null, lambda tracking) are clearly marked non-runnable.

All literal seeds in this file are dev seeds (>= 1000).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # matches the other test modules' src imports
CONFIG_DIR = REPO_ROOT / "configs"
RUN_PLAN = REPO_ROOT / "docs" / "wp22-run-plan.md"


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_module("wp22_sweep_module", "scripts/sweep.py")

WP22_CONFIGS = sorted(CONFIG_DIR.glob("wp22_*.yaml"))
WP23_CONFIG = CONFIG_DIR / "wp23_lambda_tracking.yaml"
ALL_CONFIGS = WP22_CONFIGS + [WP23_CONFIG]

# Group membership by filename prefix (mirrors docs/wp22-run-plan.md,
# Gate-1-amended matrix: reports/gate1-decision.md A1/A2/A5).
EVAL_PREFIXES = (
    "wp22_headtohead_",
    "wp22_tuneB_",
    "wp22_null_",
    "wp22_channel_",
    "wp22_goscconst_",
    "wp22_exploratory_",
)
PLACEHOLDER_CONFIGS = {  # TBD-STAGE-A: filled by scripts/plan_wp22.py
    "wp22_tuneB_muon.yaml",
    "wp22_tuneB_routed.yaml",
    "wp22_null_muon_wd.yaml",
}
BLOCKED_CONFIGS = {  # NOT-RUNNABLE-YET: need harness work before launch
    "wp22_null_muon_lrclip.yaml",
    "wp23_lambda_tracking.yaml",
}

# Gate-1 amendment A1: the oscillation-only scope, as config contracts.
OSC_ONLY_ROUTED_CONFIGS = {  # enable_noise_channel must be false
    "wp22_headtohead_routed.yaml",
    "wp22_tuneA_routed.yaml",
    "wp22_tuneB_routed.yaml",
    "wp22_null_routed_randomgating.yaml",
    "wp22_stress_routed.yaml",
    "wp22_goscconst_025.yaml",
    "wp22_goscconst_050.yaml",
    "wp22_goscconst_075.yaml",
    "wp22_beta09_oscarm.yaml",
}
EXPLORATORY_CONFIGS = {  # demoted arms: no pre-registered claim
    "wp22_exploratory_fullrouted.yaml",
    "wp22_channel_noise_only.yaml",
    "wp22_null_routed_rhoignored.yaml",
}
GOSCCONST_VALUES = {  # Gate-1 amendment A2: fixed-gain arms
    "wp22_goscconst_025.yaml": 0.25,
    "wp22_goscconst_050.yaml": 0.5,
    "wp22_goscconst_075.yaml": 0.75,
}


def load(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def test_expected_config_set_present():
    names = {p.name for p in WP22_CONFIGS}
    assert names == {
        "wp22_headtohead_muon.yaml",
        "wp22_headtohead_routed.yaml",  # Gate-1 A1: osc-only primary arm
        "wp22_headtohead_dynmuon.yaml",
        "wp22_headtohead_adamuon.yaml",
        "wp22_headtohead_normuon.yaml",
        "wp22_tuneA_muon.yaml",
        "wp22_tuneA_routed.yaml",
        "wp22_tuneB_muon.yaml",
        "wp22_tuneB_routed.yaml",
        "wp22_null_muon_wd.yaml",
        "wp22_null_muon_lrclip.yaml",
        "wp22_null_routed_rhoignored.yaml",
        "wp22_null_routed_randomgating.yaml",
        # Gate-1 A2: constant-attenuation arms.
        "wp22_goscconst_025.yaml",
        "wp22_goscconst_050.yaml",
        "wp22_goscconst_075.yaml",
        # Gate-1 A1: exploratory appendix arms (former full-routing
        # head-to-head + noise-only); wp22_channel_osc_only.yaml was DELETED
        # (byte-duplicate of the amended osc-only primary).
        "wp22_exploratory_fullrouted.yaml",
        "wp22_channel_noise_only.yaml",
        "wp22_stress_muon.yaml",
        "wp22_stress_routed.yaml",
        # Gate-1 A5: beta-sensitivity dev probe.
        "wp22_beta09_oscarm.yaml",
    }
    assert WP23_CONFIG.exists()


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_config_parses_and_is_airbench_smoke(path):
    config = load(path)
    assert isinstance(config, dict)
    assert config["experiment"] == "airbench_smoke"
    assert config["device"] == "cuda"
    assert isinstance(config.get("sweep"), dict)


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_optimizer_name_resolves_in_registry(path):
    from src.optim.registry import OPTIMIZER_REGISTRY

    name = load(path)["optimizer"]["name"]
    assert name in OPTIMIZER_REGISTRY, f"{path.name}: unknown optimizer {name!r}"


# ------------------------------------------------------------- seed hygiene


def _seed_literals(obj, path="", under_seed_key=False):
    """Every literal int under a seed-like key (count keys exempt)."""
    hits = []
    if isinstance(obj, bool):
        return hits
    if isinstance(obj, int):
        if under_seed_key:
            hits.append((path, obj))
        return hits
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            child = f"{path}.{key}" if path else str(key)
            if key_l in sweep.SEED_COUNT_KEY_EXEMPTIONS:
                under = False
            else:
                under = under_seed_key or "seed" in key_l
            hits.extend(_seed_literals(value, child, under))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            hits.extend(_seed_literals(value, f"{path}[{i}]", under_seed_key))
    return hits


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_no_literal_seed_below_1000(path):
    config = load(path)
    bad = [(where, val) for where, val in _seed_literals(config) if val < 1000]
    assert not bad, f"{path.name}: literal seed(s) < 1000: {bad}"
    # And the stricter launch-time gate must also pass (no eval literals 0-99).
    sweep.refuse_eval_seed_literals(config, path.name)


# ------------------------------------------------------- seed policy per group


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_seed_policy_matches_group(path):
    policy_spec = load(path)["sweep"]["seeds"]
    policy, seeds = sweep.resolve_seed_policy(policy_spec)
    name = path.name
    if name.startswith(EVAL_PREFIXES):
        assert policy == "eval" and seeds == list(range(100)), (
            f"{name}: comparison-table config must use the full eval policy"
        )
    elif name.startswith("wp22_tuneA_"):
        assert policy == "dev" and seeds == list(range(1000, 1025)), (
            f"{name}: stage A is dev n=25 (selection stage)"
        )
    elif name.startswith("wp22_stress_"):
        assert policy == "dev" and seeds == list(range(1000, 1010)), (
            f"{name}: LR stress is dev n=10 per point"
        )
    elif name == "wp22_beta09_oscarm.yaml":
        assert policy == "dev" and seeds == list(range(1000, 1010)), (
            f"{name}: beta sensitivity is a dev n=10 probe (Gate-1 A5)"
        )
    elif name == "wp23_lambda_tracking.yaml":
        assert policy == "dev" and seeds == list(range(1000, 1003)), (
            f"{name}: lambda tracking is dev n=3 (measurement)"
        )
    else:  # pragma: no cover - guarded by test_expected_config_set_present
        pytest.fail(f"{name}: not assigned to any group")


# --------------------------------------------- run-plan manifest cross-check


def run_plan_manifest() -> dict:
    text = RUN_PLAN.read_text()
    blocks = re.findall(r"```json\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) == 1, "docs/wp22-run-plan.md must carry exactly one fenced JSON manifest"
    return json.loads(blocks[0])


def test_manifest_covers_exactly_the_wp22_configs():
    manifest = run_plan_manifest()
    listed = set()
    for group in manifest["groups"]:
        listed.update(Path(p).name for p in group["configs"])
    assert listed == {p.name for p in WP22_CONFIGS}
    pending = manifest["pending"]
    assert set(Path(p).name for p in pending["configs"]) == {WP23_CONFIG.name}


def test_grid_counts_and_totals_match_run_plan():
    manifest = run_plan_manifest()
    total_runs = 0
    for group in manifest["groups"]:
        group_runs = 0
        for cfg_rel, declared_variants in group["configs"].items():
            config = load(REPO_ROOT / cfg_rel)
            policy, seeds = sweep.resolve_seed_policy(config["sweep"]["seeds"])
            assert policy == group["seed_policy"], f"{cfg_rel}: policy != manifest"
            assert len(seeds) == group["n_seeds"], f"{cfg_rel}: n_seeds != manifest"
            variants = sweep.expand_grid(config["sweep"].get("grid") or {})
            assert len(variants) == declared_variants, (
                f"{cfg_rel}: grid expands to {len(variants)} variants, "
                f"manifest declares {declared_variants}"
            )
            group_runs += len(variants) * len(seeds)
        assert group_runs == group["runs"], f"{group['group']}: runs mismatch"
        total_runs += group_runs
    assert total_runs == manifest["total_runs"] == 2120  # Gate-1-amended matrix
    # Budget arithmetic stays consistent with the declared assumptions.
    hours = total_runs * manifest["sec_per_run"] / 3600.0
    assert abs(hours - manifest["total_gpu_hours"]) < 0.05
    assert abs(hours * manifest["price_per_hour_usd"] - manifest["total_cost_usd"]) < 0.05


def test_tuning_grids_are_3x3_lr_x_wd_and_stress_is_lr_only():
    for name in ("wp22_tuneA_muon.yaml", "wp22_tuneA_routed.yaml"):
        grid = load(CONFIG_DIR / name)["sweep"]["grid"]
        assert grid["optimizer.lr"] == [0.12, 0.24, 0.48], name
        assert grid["optimizer.weight_decay"] == [0.0, 0.004, 0.008], name
        assert set(grid) == {"optimizer.lr", "optimizer.weight_decay"}, name
    for name in ("wp22_stress_muon.yaml", "wp22_stress_routed.yaml"):
        grid = load(CONFIG_DIR / name)["sweep"]["grid"]
        assert grid["optimizer.lr"] == [0.24, 0.36, 0.48], name  # 1x, 1.5x, 2x record
        assert set(grid) == {"optimizer.lr"}, name


# ------------------------------------------- Gate-1 amendment contracts


def test_osc_only_scope_flags(  # Gate-1 A1
):
    for name in sorted(OSC_ONLY_ROUTED_CONFIGS):
        opt = load(CONFIG_DIR / name)["optimizer"]
        assert opt["name"] == "routed", name
        assert opt["enable_noise_channel"] is False, (
            f"{name}: primary-scope routed config must be oscillation-only"
        )
        assert opt["enable_oscillation_channel"] is True, name


def test_exploratory_arms_marked_and_full_scope():  # Gate-1 A1
    for name in sorted(EXPLORATORY_CONFIGS):
        path = CONFIG_DIR / name
        assert "EXPLORATORY" in path.read_text(), (
            f"{name}: demoted arm must carry the EXPLORATORY header marker"
        )
        opt = load(path)["optimizer"]
        # The exploratory arms are the ones that still exercise the noise
        # channel (no pre-registered claim).
        assert opt["enable_noise_channel"] is True, name
    # And no primary-scope config is accidentally marked exploratory.
    for name in sorted(OSC_ONLY_ROUTED_CONFIGS):
        assert "EXPLORATORY" not in (CONFIG_DIR / name).read_text(), name


def test_goscconst_arms_carry_fixed_gain():  # Gate-1 A2
    for name, value in sorted(GOSCCONST_VALUES.items()):
        opt = load(CONFIG_DIR / name)["optimizer"]
        assert opt["g_osc_const"] == pytest.approx(value), name
        assert opt["enable_oscillation_channel"] is True, name
    # The adaptive primary arm must NOT set g_osc_const.
    for name in ("wp22_headtohead_routed.yaml", "wp22_tuneA_routed.yaml"):
        assert "g_osc_const" not in load(CONFIG_DIR / name)["optimizer"], (
            f"{name}: adaptive arm must not fix g_osc_const"
        )


def test_beta_sensitivity_probe_is_beta09():  # Gate-1 A5
    opt = load(CONFIG_DIR / "wp22_beta09_oscarm.yaml")["optimizer"]
    assert opt["beta"] == pytest.approx(0.9)
    reference = load(CONFIG_DIR / "wp22_headtohead_routed.yaml")["optimizer"]
    assert reference["beta"] == pytest.approx(0.99)
    # Identical to the primary arm apart from beta (and nothing else).
    assert {k: v for k, v in opt.items() if k != "beta"} == {
        k: v for k, v in reference.items() if k != "beta"
    }


def test_tuneA_routed_grid_matches_muon_but_osc_only():  # Gate-1 A1
    routed = load(CONFIG_DIR / "wp22_tuneA_routed.yaml")
    muon = load(CONFIG_DIR / "wp22_tuneA_muon.yaml")
    assert routed["sweep"]["grid"] == muon["sweep"]["grid"], (
        "stage-A grid must stay identical across optimizers (equal tuning "
        "effort); only the routed rows' channel scope changed"
    )


# -------------------------------------------------- non-runnable markers


def test_stage_b_placeholders_are_marked_and_unrunnable():
    """Stage-B configs are either unfilled placeholders or properly filled.

    Unfilled: PLACEHOLDER status + TBD-STAGE-A optimizer scalars (unrunnable).
    Filled: a status recording the fill provenance and fully numeric
    optimizer scalars — no TBD strings may remain.
    """
    for name in sorted(PLACEHOLDER_CONFIGS):
        config = load(CONFIG_DIR / name)
        status = str(config.get("status", ""))
        tbd_values = [v for v in config["optimizer"].values() if v == "TBD-STAGE-A"]
        if status.startswith("PLACEHOLDER-TBD-STAGE-A"):
            assert tbd_values, f"{name}: placeholder must carry TBD-STAGE-A value(s)"
        else:
            assert status.startswith("filled"), (
                f"{name}: status must be PLACEHOLDER-TBD-STAGE-A or record the fill"
            )
            assert not tbd_values, f"{name}: filled config still carries TBD values"


def test_blocked_configs_are_marked_not_runnable_yet():
    for name in sorted(BLOCKED_CONFIGS):
        config = load(CONFIG_DIR / name)
        status = str(config.get("status", ""))
        assert status.startswith("NOT-RUNNABLE-YET"), f"{name}: missing NOT-RUNNABLE-YET marker"


def test_runnable_configs_carry_no_status_marker_or_tbd():
    unrunnable = PLACEHOLDER_CONFIGS | BLOCKED_CONFIGS
    for path in ALL_CONFIGS:
        if path.name in unrunnable:
            continue
        config = load(path)
        assert "status" not in config, f"{path.name}: unexpected status marker"
        assert "TBD" not in path.read_text().replace("TBD-TUNABLE", "").replace(
            "TBD-CHECK-GATE1", ""
        ), f"{path.name}: unexplained TBD value in a runnable config"


def test_grad_clip_null_still_requires_harness_addition():
    """wp22_null_muon_lrclip assumes recipe.grad_clip, which airbench_zoo.py
    does not implement yet (unknown recipe keys are silently ignored, so an
    early launch would look valid while running unclipped). If this test
    fails because grad_clip landed in the harness, remove the config's
    NOT-RUNNABLE-YET marker and this test together (human-verified)."""
    harness = (REPO_ROOT / "src" / "optim" / "airbench_zoo.py").read_text()
    assert "grad_clip" not in harness, (
        "harness now knows grad_clip: update wp22_null_muon_lrclip.yaml's "
        "status marker and this test"
    )
    config = load(CONFIG_DIR / "wp22_null_muon_lrclip.yaml")
    assert config["recipe"]["grad_clip"] == 1.0
    assert config["optimizer"]["lr"] == pytest.approx(0.36)  # 1.5x record


# ------------------------------------------------------ recipe uniformity


def test_identical_recipe_and_training_setup_across_matrix():
    reference = load(CONFIG_DIR / "wp22_headtohead_muon.yaml")
    for path in ALL_CONFIGS:
        config = load(path)
        assert config["train"] == reference["train"], f"{path.name}: train block differs"
        assert config["data"] == reference["data"], f"{path.name}: data block differs"
        recipe = dict(config["recipe"])
        # The clipping ablation's whole point is its grad_clip key; everything
        # else must match the shared recipe exactly.
        recipe.pop("grad_clip", None)
        assert recipe == reference["recipe"], f"{path.name}: recipe differs"


def test_all_configs_expand_through_sweep_tooling():
    """Launch-path smoke: expansion (incl. eval-literal refusal and variant
    naming) succeeds for every config; placeholders expand too (they are
    blocked semantically by markers, not by the expander)."""
    for path in ALL_CONFIGS:
        plan = sweep.expand_sweep(load(path), path)
        assert plan["runs"], path.name
