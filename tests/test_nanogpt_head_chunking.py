"""PORT CHANGE P5: chunked lm_head + loss must equal the full-width path.

The record computes one full-width head GEMM, fp32 soft-cap, and a single
cross_entropy (sum-reduced in train, mean in eval). The chunked path computes
the same quantities over row chunks (train chunks gradient-checkpointed).
These tests pin the equivalence — losses match to fp32 tolerance in both
modes, and so do gradients through the checkpointed train path — plus the
config wiring (flag, faithfulness, divisibility validation).
"""

import copy

import pytest
import torch

from src.nanogpt.config import ConfigError, NanoGPTConfig
from src.nanogpt.model import GPT


def _tiny(head_chunk_rows=None, seed=0):
    torch.manual_seed(seed)
    return GPT(
        vocab_size=640, num_layers=12, num_heads=1, model_dim=128,
        max_seq_len=512, world_size=8, use_fp8=False, attention_impl="sdpa",
        head_chunk_rows=head_chunk_rows,
    )


def _batch(seed=1, n=512):
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, 640, (n,), generator=g)
    # the record derives block masks from BOS positions; plant a few
    inputs[0] = 50256 % 640
    targets = torch.randint(0, 640, (n,), generator=g)
    return inputs, targets


def _clone_from(src_model, head_chunk_rows):
    dst = _tiny(head_chunk_rows=head_chunk_rows)
    dst.load_state_dict(copy.deepcopy(src_model.state_dict()))
    return dst


@pytest.mark.parametrize("chunk", [64, 128, 256])
def test_train_loss_matches_full_width(chunk):
    full = _tiny()
    chunked = _clone_from(full, chunk)
    full.train()
    chunked.train()
    inputs, targets = _batch()
    sw = torch.tensor(1)
    loss_full = full(inputs, targets, sw)
    loss_chunk = chunked(inputs, targets, sw)
    assert loss_chunk.item() == pytest.approx(loss_full.item(), rel=1e-5)


def test_eval_mean_matches_full_width():
    full = _tiny()
    chunked = _clone_from(full, 128)
    full.eval()
    chunked.eval()
    inputs, targets = _batch()
    sw = torch.tensor(1)
    with torch.no_grad():
        loss_full = full(inputs, targets, sw)
        loss_chunk = chunked(inputs, targets, sw)
    assert loss_chunk.item() == pytest.approx(loss_full.item(), rel=1e-6)


def test_gradients_match_through_checkpointed_chunks():
    full = _tiny()
    chunked = _clone_from(full, 128)
    full.train()
    chunked.train()
    inputs, targets = _batch()
    sw = torch.tensor(1)
    full(inputs, targets, sw).backward()
    chunked(inputs, targets, sw).backward()
    for (n1, p1), (n2, p2) in zip(
        full.named_parameters(), chunked.named_parameters()
    ):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None, n1
            continue
        assert torch.allclose(
            p1.grad.float(), p2.grad.float(), rtol=1e-4, atol=1e-6
        ), f"grad mismatch: {n1}"


def test_non_divisible_rows_assert():
    m = _tiny(head_chunk_rows=100)  # 512 % 100 != 0
    m.train()
    inputs, targets = _batch()
    with pytest.raises(AssertionError):
        m(inputs, targets, torch.tensor(1))


def _cfg(**kw):
    return NanoGPTConfig(
        **kw
    )


class TestConfigWiring:
    def test_flag_off_by_default_and_faithfulness(self):
        base = _cfg(device_count=8)
        assert base.head_chunk_rows is None
        assert "head_chunk_rows" not in base.deviations()
        chunked = _cfg(device_count=8, head_chunk_rows=8192)
        assert not chunked.record_faithful
        assert "head_chunk_rows" in chunked.deviations()

    def test_divisibility_validated(self):
        with pytest.raises(ConfigError):
            _cfg(head_chunk_rows=10_000)  # divides neither 49152 nor 262144
        cfg = _cfg(head_chunk_rows=8192)  # divides both record seq lens
        assert cfg.train_seq_len % 8192 == 0 and cfg.val_seq_len % 8192 == 0

    def test_positive_validated(self):
        with pytest.raises(ConfigError):
            _cfg(head_chunk_rows=0)


class TestSeedPlumbing:
    """Regression: sweep-materialized configs carry no seed key; the CLI seed
    must reach NanoGPTConfig (trainer re-seed + seed-keyed checkpoint path)."""

    def test_from_config_missing_seed_falls_back_to_default(self):
        # the failure mode: no top-level seed -> dataclass default
        cfg = NanoGPTConfig.from_config({"nanogpt": {}})
        assert cfg.seed == 1000

    def test_run_py_injects_cli_seed_into_config(self, tmp_path, monkeypatch):
        import importlib.util, json, yaml
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "rm_run_seedtest", Path(__file__).resolve().parent.parent / "scripts" / "run.py"
        )
        run_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mod)

        seen = {}

        def fake_experiment(config, device):
            seen["config_seed"] = config.get("seed")
            return {"ok": True}

        monkeypatch.setitem(run_mod.EXPERIMENT_REGISTRY, "smoke", fake_experiment)
        cfg_file = tmp_path / "sweepstyle.yaml"  # no seed key, like sweep output
        cfg_file.write_text(yaml.safe_dump({"experiment": "smoke", "device": "cpu"}))
        rc = run_mod.main([str(cfg_file), "--seed", "1717", "--out-dir", str(tmp_path)])
        assert rc == 0
        assert seen["config_seed"] == 1717
        out = list(tmp_path.glob("smoke_seed1717_*.json"))
        assert len(out) == 1
        written = json.loads(out[0].read_text())
        assert written["seed"] == 1717
        assert written["config"]["contents"]["seed"] == 1717


class TestChunksPerStep:
    """Program #7 batch axis: chunks_per_step changes tokens/step exactly."""

    def test_default_is_record(self):
        cfg = _cfg(device_count=8)
        assert cfg.effective_chunks == 8
        assert cfg.tokens_per_step == 393_216
        assert "chunks_per_step" not in cfg.deviations()

    def test_smaller_batch(self):
        cfg = _cfg(device_count=1, chunks_per_step=2)
        assert cfg.effective_chunks == 2
        assert cfg.accum_factor == 2
        assert cfg.tokens_per_step == 2 * 49_152
        assert not cfg.record_faithful
        assert "chunks_per_step" in cfg.deviations()

    def test_larger_batch(self):
        cfg = _cfg(device_count=1, chunks_per_step=16)
        assert cfg.accum_factor == 16
        assert cfg.tokens_per_step == 16 * 49_152

    def test_explicit_record_value_is_faithful_axis(self):
        cfg = _cfg(device_count=8, chunks_per_step=8)
        assert "chunks_per_step" not in cfg.deviations()
        assert cfg.record_faithful

    def test_device_count_must_divide(self):
        with pytest.raises(ConfigError):
            _cfg(device_count=2, chunks_per_step=3)
        cfg = _cfg(device_count=2, chunks_per_step=4)
        assert cfg.accum_factor == 2

    def test_positive(self):
        with pytest.raises(ConfigError):
            _cfg(device_count=1, chunks_per_step=0)


def test_generator_generic_in_world_size(tmp_path):
    """The data generator yields exactly chunks/device micro-batches per step
    for a non-record world size (program #7 relies on this)."""
    import numpy as np
    from src.nanogpt.data import RecordDataGenerator

    tokens = np.zeros(120_000, dtype=np.uint16)
    tokens[::997] = 50256  # BOS markers
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = 20240520, 1, len(tokens)
    shard = tmp_path / "fineweb_train_000001.bin"
    with open(shard, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.tobytes())

    gen = RecordDataGenerator(
        str(tmp_path / "fineweb_train_*.bin"),
        local_batch_size=4096,
        record_world_size=2,          # program-#7 small-batch geometry
        device_count=1,
        rank=0,
        align_to_bos=True,
        device=torch.device("cpu"),
    )
    micro = list(gen.next_step())
    assert len(micro) == 2  # 2 chunks / 1 device
    for inputs, targets in micro:
        assert inputs.numel() == 4096


class TestBatchSpanFloor:
    """Program #7 loader fix: the BOS search window is floored at the
    record's window so small-chunk arms survive sparse-BOS stretches."""

    def _gen(self, tmp_path, world, n_tokens=900_000, bos_stride=997):
        import numpy as np
        from src.nanogpt.data import RecordDataGenerator

        tokens = np.zeros(n_tokens, dtype=np.uint16)
        tokens[::bos_stride] = 50256
        header = np.zeros(256, dtype=np.int32)
        header[0], header[1], header[2] = 20240520, 1, len(tokens)
        shard = tmp_path / "fineweb_train_000001.bin"
        with open(shard, "wb") as fh:
            fh.write(header.tobytes())
            fh.write(tokens.tobytes())
        return RecordDataGenerator(
            str(tmp_path / "fineweb_train_*.bin"),
            local_batch_size=49_152,
            record_world_size=world,
            device_count=1,
            rank=0,
            align_to_bos=True,
            device=torch.device("cpu"),
        )

    def test_record_window_unchanged(self, tmp_path):
        gen = self._gen(tmp_path, world=8)
        assert gen.max_batch_span == 2 * 8 * 49_152

    def test_small_chunks_get_record_floor(self, tmp_path):
        gen = self._gen(tmp_path, world=2)
        assert gen.max_batch_span == 2 * 8 * 49_152  # floored, not 2*2*49152

    def test_large_chunks_scale_past_floor(self, tmp_path):
        gen = self._gen(tmp_path, world=16, n_tokens=3_400_000)
        assert gen.max_batch_span == 2 * 16 * 49_152

    def test_sparse_bos_stretch_survives_at_small_chunks(self, tmp_path):
        # one qualifying boundary per ~120k tokens: the pre-fix 2*2*49152
        # window (196k) holds barely 1 chunk boundary and asserted; the
        # floored window (786k) finds the 2 chunks comfortably.
        gen = self._gen(tmp_path, world=2, bos_stride=120_000)
        micro = list(gen.next_step())
        assert len(micro) == 2
