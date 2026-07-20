"""Config for the WP0.2 nanogpt port: record defaults + the accumulation math.

Every field defaults to the pinned record's own value (record source file and
line references below; see ``src/nanogpt/__init__.py`` for which log is
authoritative). A config that sets nothing but ``seed`` and ``device_count``
is the record, modulo the documented port deviations.

THE ACCUMULATION MATH (the whole point of this module)
------------------------------------------------------
The record runs one forward/backward per optimizer step on ``world_size = 8``
ranks, each rank consuming ``train_seq_len = 48*1024 = 49,152`` tokens
(RECORD:692 ``distributed_data_generator(..., world_size * args.seq_len, ...)``,
RECORD:579 ``seq_len = 48*1024``). So::

    tokens/optimizer-step = record_world_size * train_seq_len
                          = 8 * 49,152 = 393,216

Nothing in the record derives the batch from anything but ``world_size``, so
on D devices the token batch would silently shrink by 8/D. The port restores
it with G micro-batches per device per optimizer step::

    G = accumulation_factor = record_world_size / device_count
    tokens/step = device_count * G * train_seq_len = 393,216   (invariant)

    D = 1  ->  G = 8      D = 2  ->  G = 4      D = 8  ->  G = 1 (= the record)

Gradient scaling. The record's training loss uses ``reduction="sum"``
(RECORD:503), so each rank's backward produces the *sum* over its 49,152
tokens, and the record reduces across ranks with ``ReduceOp.AVG``
(RECORD:178 / RECORD:233). The record's effective gradient is therefore

    g_record = (1/8) * sum over all 8 chunks of (chunk gradient sum).

On D ranks accumulating G micro-batches, the raw accumulated gradient before
the AVG all-reduce is ``sum over that rank's G chunks``; the AVG over D ranks
gives ``(1/D) * sum over all 8 chunks``. Multiplying each micro-loss by 1/G
before ``.backward()`` gives ``(1/(D*G)) * sum over all 8 chunks = g_record``
exactly (gradients are linear in the loss). ``device_count * G == 8`` always,
so this is exact, and because G is a power of two the rescale is exact in
binary floating point (this also keeps the FP8 head-backward quantization
bit-comparable — see docs/nanogpt-port.md, deviation list).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The authoritative record log (see src/nanogpt/__init__.py for why this file
# and not the "primary" log cited in reports/wp02-record-candidates.md).
RECORD_LOG = (
    REPO_ROOT
    / "vendor/modded-nanogpt/records/track_1_short/2025-07-12_BosAlign"
    / "0c5449cc-0b01-4ecc-bec3-f46a09741d60.txt"
)
RECORD_DIR = RECORD_LOG.parent
# Number of script lines before the log body starts (the "="*100 separator).
RECORD_SCRIPT_LINES = 783

RECORD_WORLD_SIZE = 8  # RECORD: run under `torchrun --nproc_per_node=8`
RECORD_TRAIN_SEQ_LEN = 48 * 1024  # RECORD:579
RECORD_TOKENS_PER_STEP = RECORD_WORLD_SIZE * RECORD_TRAIN_SEQ_LEN  # 393,216
RECORD_NUM_ITERATIONS = 1750  # RECORD:574
RECORD_TARGET_VAL_LOSS = 3.28  # vendor/modded-nanogpt/README.md:3
# The record's own n=20 validation distribution (record README, 20 runs).
RECORD_SEED_DISTRIBUTION = (
    3.2784, 3.2791, 3.2819, 3.2801, 3.2788, 3.2794, 3.2782, 3.2770, 3.2784,
    3.2773, 3.2803, 3.2792, 3.2792, 3.2804, 3.2817, 3.2805, 3.2783, 3.2789,
    3.2779, 3.2778,
)


class ConfigError(ValueError):
    """Raised for a config that cannot be run faithfully."""


def accumulation_factor(device_count: int, record_world_size: int = RECORD_WORLD_SIZE) -> int:
    """Micro-batches per device per optimizer step. See module docstring.

    Raises :class:`ConfigError` unless ``device_count`` divides
    ``record_world_size`` — a non-integer accumulation factor cannot reproduce
    the record's token batch, and silently rounding it would corrupt the whole
    benchmark.
    """
    if device_count < 1:
        raise ConfigError(f"device_count must be >= 1, got {device_count}")
    if record_world_size % device_count != 0:
        raise ConfigError(
            f"device_count {device_count} does not divide the record's "
            f"world_size {record_world_size}; the record's token batch "
            f"({record_world_size} x seq_len) cannot be reproduced exactly. "
            "Use a device count in {1, 2, 4, 8}."
        )
    return record_world_size // device_count


def tokens_per_step(
    device_count: int,
    train_seq_len: int = RECORD_TRAIN_SEQ_LEN,
    record_world_size: int = RECORD_WORLD_SIZE,
) -> int:
    """Tokens per optimizer step under the port — invariant in device_count."""
    return device_count * accumulation_factor(device_count, record_world_size) * train_seq_len


@dataclass
class CheckpointConfig:
    """Spot-tier resume (docs/hyperstack-runbook.md 'Checkpoint-resume').

    Checkpoints are VM-local and never sync; only the metrics JSON does.
    ``every_steps: 0`` disables checkpointing entirely.
    """

    dir: str = "checkpoints"
    every_steps: int = 0
    resume: bool = True


@dataclass
class NanoGPTConfig:
    """All knobs. Defaults == the pinned record; deviations are flagged."""

    # ---- data (RECORD:570-572) -------------------------------------------
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760  # RECORD:572 — fixed for comparability
    train_seq_len: int = RECORD_TRAIN_SEQ_LEN  # RECORD:579
    val_seq_len: int = 4 * 64 * 1024  # RECORD:580
    train_align_to_bos: bool = True  # RECORD:581
    val_align_to_bos: bool = False  # RECORD:582

    # ---- optimization schedule (RECORD:574-575, 670-684) -----------------
    num_iterations: int = RECORD_NUM_ITERATIONS  # RECORD:574 (config-driven: PORT)
    cooldown_frac: float = 0.45  # RECORD:575
    min_lr_frac: float = 0.05  # RECORD:674 (`(1 - w) * 0.05`)
    momentum_warmup_steps: int = 300  # RECORD:769 (`min(step / 300, 1)`)
    momentum_start: float = 0.85  # RECORD:771
    momentum_end: float = 0.95  # RECORD:771
    window_warmup_max: int = 1728  # RECORD:683 (`next_multiple_of_n(1728 * x, n=128)`)

    # ---- model (RECORD:627) ----------------------------------------------
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 6
    model_dim: int = 768

    # ---- optimizers (RECORD:644-645) — instantiation values, not defaults -
    muon_lr: float = 0.05
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    adam_lr: float = 0.008
    adam_betas: Tuple[float, float] = (0.8, 0.95)
    adam_eps: float = 1e-10
    adam_weight_decay: float = 0.0

    # ---- eval / logging (RECORD:576) -------------------------------------
    val_loss_every: int = 125
    target_val_loss: float = RECORD_TARGET_VAL_LOSS

    # ---- PORT-ONLY knobs (each one a documented deviation) ---------------
    device_count: int = 1
    record_world_size: int = RECORD_WORLD_SIZE
    seed: int = 1000  # PORT: the record seeds nothing; we seed for reproducibility
    warmup_steps: int = 10  # RECORD:689 (kernel warmup; no ML effect)
    compile: bool = True  # RECORD:686 `torch.compile(model, dynamic=False)`
    precision_mode: str = "fp8"  # "fp8" == record. "bf16" == NOT record-faithful
    attention_impl: str = "flex"  # "flex" == record. "sdpa" == NOT record-faithful
    max_steps: Optional[int] = None  # PORT: early stop for smoke runs (not a run of record)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # ------------------------------------------------------------------ api
    def __post_init__(self) -> None:
        if isinstance(self.checkpoint, dict):
            self.checkpoint = CheckpointConfig(**self.checkpoint)
        if isinstance(self.adam_betas, list):
            self.adam_betas = tuple(self.adam_betas)
        self.validate()

    @property
    def accum_factor(self) -> int:
        """G — micro-batches per device per optimizer step."""
        return accumulation_factor(self.device_count, self.record_world_size)

    @property
    def tokens_per_step(self) -> int:
        return tokens_per_step(self.device_count, self.train_seq_len, self.record_world_size)

    @property
    def val_chunks(self) -> int:
        """Total validation chunks per eval (record: 8 ranks x 5 steps = 40)."""
        return self.val_tokens // self.val_seq_len

    @property
    def record_faithful(self) -> bool:
        """False if any *numerics-changing* deviation flag is set."""
        return (
            self.precision_mode == "fp8"
            and self.attention_impl == "flex"
            and self.min_lr_frac == 0.05
            and self.num_iterations == RECORD_NUM_ITERATIONS
            and self.train_seq_len == RECORD_TRAIN_SEQ_LEN
            and self.tokens_per_step == RECORD_TOKENS_PER_STEP
            and self.max_steps is None
        )

    def deviations(self) -> Dict[str, str]:
        """Human-readable map of active deviations from the record."""
        out: Dict[str, str] = {}
        if self.device_count != self.record_world_size:
            out["grad_accumulation"] = (
                f"{self.device_count} device(s) x {self.accum_factor} micro-batches "
                f"= {self.record_world_size} record chunks per optimizer step; "
                f"tokens/step {self.tokens_per_step} (record {RECORD_TOKENS_PER_STEP})"
            )
        if self.precision_mode != "fp8":
            out["precision_mode"] = (
                f"lm_head runs in {self.precision_mode}, NOT the record's FP8 "
                "(torch._scaled_mm float8_e4m3fn). NOT RECORD-FAITHFUL: changes "
                "head numerics; use only on non-H100-class GPUs."
            )
        if self.attention_impl != "flex":
            out["attention_impl"] = (
                f"attention uses {self.attention_impl}, NOT the record's "
                "FlexAttention block masks. NOT RECORD-FAITHFUL: the sliding "
                "window is applied at block granularity by a dense mask. "
                "CPU-test path only."
            )
        if self.num_iterations != RECORD_NUM_ITERATIONS:
            out["num_iterations"] = f"{self.num_iterations} (record {RECORD_NUM_ITERATIONS})"
        if self.min_lr_frac != 0.05:
            out["min_lr_frac"] = f"{self.min_lr_frac} (record 0.05)"
        if self.max_steps is not None:
            out["max_steps"] = f"truncated at {self.max_steps} steps — smoke run, not a record run"
        if not self.compile:
            out["compile"] = "torch.compile disabled (record compiles); slower, ML-neutral"
        return out

    def validate(self) -> None:
        if self.precision_mode not in ("fp8", "bf16"):
            raise ConfigError(f"precision_mode must be 'fp8' or 'bf16', got {self.precision_mode!r}")
        if self.attention_impl not in ("flex", "sdpa"):
            raise ConfigError(f"attention_impl must be 'flex' or 'sdpa', got {self.attention_impl!r}")
        accumulation_factor(self.device_count, self.record_world_size)  # raises
        if self.tokens_per_step != self.record_world_size * self.train_seq_len:
            raise ConfigError("token batch per optimizer step does not match the record")
        if self.val_tokens % self.val_seq_len != 0:
            raise ConfigError(
                f"val_tokens {self.val_tokens} not divisible by val_seq_len {self.val_seq_len} "
                "(RECORD:735 asserts this)"
            )
        if self.val_chunks % self.device_count != 0:
            raise ConfigError(
                f"{self.val_chunks} validation chunks do not split evenly over "
                f"{self.device_count} device(s)"
            )
        if self.train_seq_len % 128 != 0:
            raise ConfigError("train_seq_len must be a multiple of the 128-token block size")
        if self.num_layers % 2 != 0:
            raise ConfigError("num_layers must be even (RECORD:419 asserts this)")
        if self.max_steps is not None and self.max_steps < 0:
            raise ConfigError("max_steps must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["adam_betas"] = list(self.adam_betas)
        return d

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NanoGPTConfig":
        """Build from a parsed run.py YAML config (its ``nanogpt:`` block).

        The run-level ``seed`` (which ``scripts/run.py`` may override on the
        command line) wins over any ``nanogpt.seed``.
        """
        block = dict(config.get("nanogpt", {}) or {})
        if "seed" in config and config["seed"] is not None:
            block["seed"] = int(config["seed"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(block) - known
        if unknown:
            raise ConfigError(
                f"unknown nanogpt config keys: {sorted(unknown)}; known: {sorted(known)}. "
                "(Unknown keys are rejected rather than ignored — a silently "
                "dropped knob is a silent deviation from the record.)"
            )
        return cls(**block)
