"""Bit-exact snapshot/restore of live training state (flow-probe experiments).

Used by the anneal-dissection experiment (`airbench_anneal_branch`,
docs/litreview/j-theory-theorem-sweep.md §6.1) to branch several anneals off
one training trajectory: capture parameters, floating buffers (BatchNorm
running stats), and every optimizer's state tensors; restore them exactly so
each branch starts from the identical dynamical state.

Harness-agnostic and CPU-testable. Restore is in-place (`copy_`) so compiled
models and fused optimizers keep their tensor identities.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch


@torch.no_grad()
def snapshot_training_state(model, optimizers) -> Dict[str, Any]:
    """Capture params, floating buffers, and optimizer states as clones."""
    return {
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "buffers": {
            n: b.detach().clone()
            for n, b in model.named_buffers()
            if b.is_floating_point()
        },
        # state_dict tensors alias live optimizer state; deep-copy to freeze.
        "optimizers": [copy.deepcopy(opt.state_dict()) for opt in optimizers],
    }


@torch.no_grad()
def restore_training_state(model, optimizers, snap: Dict[str, Any]) -> None:
    """Restore a snapshot taken on the SAME model/optimizers, in place."""
    params = dict(model.named_parameters())
    if set(params) != set(snap["params"]):
        raise SystemExit("snapshot/model parameter names do not match")
    for name, saved in snap["params"].items():
        params[name].data.copy_(saved)
    buffers = {n: b for n, b in model.named_buffers() if b.is_floating_point()}
    if set(buffers) != set(snap["buffers"]):
        raise SystemExit("snapshot/model buffer names do not match")
    for name, saved in snap["buffers"].items():
        buffers[name].copy_(saved)
    if len(optimizers) != len(snap["optimizers"]):
        raise SystemExit("snapshot/optimizer count mismatch")
    for opt, saved in zip(optimizers, snap["optimizers"]):
        # load_state_dict keeps live param references; deep-copy again so a
        # later re-restore from the same snapshot stays pristine.
        opt.load_state_dict(copy.deepcopy(saved))


def snapshot_equal(model, optimizers, snap: Dict[str, Any]) -> bool:
    """True iff the live state matches the snapshot exactly (test helper)."""
    for name, p in model.named_parameters():
        if not torch.equal(p.detach(), snap["params"][name]):
            return False
    for name, b in model.named_buffers():
        if b.is_floating_point() and not torch.equal(b.detach(), snap["buffers"][name]):
            return False
    for opt, saved in zip(optimizers, snap["optimizers"]):
        live: List = list(opt.state_dict()["state"].items())
        for (k, states), (k2, saved_states) in zip(live, saved["state"].items()):
            if k != k2 or set(states) != set(saved_states):
                return False
            for field, value in states.items():
                sv = saved_states[field]
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, sv):
                        return False
                elif value != sv:
                    return False
    return True
