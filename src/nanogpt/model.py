"""The record's model, copied VERBATIM except for the marked port changes.

Record source (see src/nanogpt/__init__.py): the 2025-07-12_BosAlign
validation script, ``0c5449cc-....txt`` lines 27-104 (FP8 ops) and 289-504
(modules). ``RECORD:<n>`` below is a line number in that file.

PORT CHANGES IN THIS FILE — the complete list (nothing else differs):

P1. ``GPT.__init__`` takes ``use_fp8`` (default True = record) so the LM head
    can fall back to bf16 on non-FP8 GPUs.  RECORD:415 hardcodes
    ``use_fp8=True``.  **The bf16 path is NOT RECORD-FAITHFUL** — it changes
    head numerics.  Config: ``precision_mode``.
P2. ``GPT.__init__`` takes ``world_size`` for the ``scalars`` padding instead
    of reading the process world size (RECORD:420 ``pad = (-num_layers * 5) %
    world_size``).  The port passes the **record's** world size (8), so the
    parameter tensor is bit-identical to the record's (64 entries, of which
    the last 4 are unused padding) at any device count.
P3. ``create_blockmasks`` takes the device from ``input_seq`` instead of
    hardcoding ``device="cuda"`` (RECORD:447).  No numerical change on CUDA.
P4. ``attention_impl="sdpa"`` adds a dense-mask fallback used **only** so the
    model can be forward/backward-tested on CPU (FlexAttention has no CPU
    backward).  **NOT RECORD-FAITHFUL**; refused for real runs by
    ``src/nanogpt/train.py`` unless explicitly configured.
P5. The record executes ``torch.empty(1, device="cuda", ...).backward()`` at
    import (RECORD:15, "prevents a bug on some systems"); the port does it in
    ``train.py`` at startup so this module imports on a CPU box.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention

# --------------------------------------------------------------------------
# RECORD:26-105 — Custom operators: FP8 matmul by @YouJiacheng — verbatim
# --------------------------------------------------------------------------


@torch.library.custom_op("routed_muon_nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)


@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@torch.library.custom_op("routed_muon_nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)


@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.routed_muon_nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None


def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)


# --------------------------------------------------------------------------
# RECORD:288-398 — model modules — verbatim except where marked
# --------------------------------------------------------------------------


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.routed_muon_nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        if isinstance(block_mask, BlockMask):
            y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        else:
            # PORT CHANGE P4 (NOT RECORD-FAITHFUL): dense-mask SDPA fallback,
            # CPU test path only. `block_mask` is a (T, T) bool tensor built by
            # GPT.create_blockmasks in "sdpa" mode.
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=block_mask[None, None], scale=self.attn_scale,
            ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        x = x + self.mlp(norm(x))
        return x


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


# --------------------------------------------------------------------------
# RECORD:400-504 — the main model — verbatim except P1/P2/P3/P4
# --------------------------------------------------------------------------


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        max_seq_len: int,
        *,
        world_size: int = 8,      # PORT CHANGE P2 (record: process world size, always 8)
        use_fp8: bool = True,     # PORT CHANGE P1 (record: hardcoded True at RECORD:415)
        attention_impl: str = "flex",  # PORT CHANGE P4 (record: always flex)
        head_chunk_rows: int | None = None,  # PORT CHANGE P5 (record: one full-width head GEMM)
    ):
        super().__init__()
        self.attention_impl = attention_impl
        # PORT CHANGE P5: compute lm_head + soft-cap + cross-entropy over row
        # chunks of this size (train: each chunk checkpointed so its logits
        # are recomputed in backward instead of saved).  The record's
        # reduction='sum' (train) and 'mean' (eval) decompose exactly over row
        # chunks and the fp8 head scales are static constants, so the result
        # is the record's loss up to fp32 summation order.  NOT RECORD-
        # FAITHFUL, but the only way the record's 49,152-token train chunk and
        # 262,144-token val sequences fit a 32 GB GPU (the full-width path
        # materializes multi-GB fp32 logits: 49,152 x 50,304 x 4 B in train,
        # 262,144 x 50,304 x 4 B in val).
        self.head_chunk_rows = head_chunk_rows
        self.embed = nn.Embedding(vocab_size, model_dim)
        for param in self.embed.parameters():
            param.lr_mul = 75.
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        for embeds in self.value_embeds:
            for param in self.value_embeds.parameters():
                param.lr_mul = 75.
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.lr_mul = 27.5
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % world_size
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
            torch.ones(pad),
        ]))
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        # PORT CHANGE P3: device from input_seq (record: device="cuda", RECORD:447).
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=input_seq.device)
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all

        if self.attention_impl != "flex":
            # PORT CHANGE P4 (NOT RECORD-FAITHFUL): dense (T, T) masks for the
            # CPU/SDPA test path. Same document-causal rule, and the sliding
            # window applied at 128-block granularity, matching the semantics
            # of the clamped from_kv_blocks construction below closely enough
            # for a smoke test — but it is NOT the record's kernel.
            return (
                self._dense_mask(docs, blockmask_any, sliding_window_num_blocks, BLOCK_SIZE),
                self._dense_mask(docs, blockmask_any, sliding_window_num_blocks // 2, BLOCK_SIZE),
            )

        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def _dense_mask(self, docs: Tensor, blockmask_any: Tensor, window_size_blocks: Tensor, block_size: int) -> Tensor:
        """PORT-ONLY (P4): dense boolean attention mask for the SDPA fallback."""
        T = docs.numel()
        if T > 8192:
            raise RuntimeError(
                f"dense SDPA mask requested for sequence length {T}; this path is "
                "for CPU smoke tests only (memory blows up). Use attention_impl='flex'."
            )
        idx = torch.arange(T, device=docs.device)
        blk = idx // block_size
        window = int(window_size_blocks)
        allowed = (
            (idx[:, None] >= idx[None, :])
            & (docs[:, None] == docs[None, :])
            & ((blk[:, None] - blk[None, :]) < max(window, 1))
        )
        # Never leave a row fully masked (SDPA produces NaNs otherwise).
        allowed[idx, idx] = True
        return allowed

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        if self.head_chunk_rows is None:
            logits = self.lm_head(x).float()
            # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
            return loss
        return self._chunked_head_loss(x, target_seq)

    def _chunked_head_loss(self, x: Tensor, target_seq: Tensor) -> Tensor:
        """PORT CHANGE P5: the record's head + soft-cap + loss over row chunks.

        Identical math to the full-width path: cross_entropy with
        reduction='sum' is additive over disjoint row chunks, and eval's
        'mean' is the same sum divided by the row count. Each train chunk is
        gradient-checkpointed so backward recomputes its logits instead of
        holding every chunk's fp32 logits simultaneously; eval runs under the
        caller's no_grad. Deviation from the record: fp32 accumulation order
        of the loss sum and of the lm_head weight-gradient (summed per chunk
        rather than in one GEMM).
        """
        x_flat = x.view(-1, x.size(-1))
        n_rows = x_flat.size(0)
        chunk = self.head_chunk_rows
        assert n_rows % chunk == 0, (
            f"sequence rows {n_rows} not divisible by head_chunk_rows {chunk}"
        )

        def chunk_loss(xc: Tensor, tc: Tensor) -> Tensor:
            logits_c = self.lm_head(xc).float()
            logits_c = 30 * torch.sigmoid(logits_c / (7.5 * xc.size(-1) ** 0.5))
            return F.cross_entropy(logits_c, tc, reduction="sum")

        total = x_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, n_rows, chunk):
            xc = x_flat[start:start + chunk]
            tc = target_seq[start:start + chunk]
            if self.training:
                total = total + torch.utils.checkpoint.checkpoint(
                    chunk_loss, xc, tc, use_reentrant=False
                )
            else:
                total = total + chunk_loss(xc, tc)
        return total if self.training else total / n_rows
