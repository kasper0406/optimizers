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
