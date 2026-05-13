"""Approximate FLOPs counter for classical PyTorch modules.

We register forward hooks on Conv2d / Linear / MHA / LayerNorm to accumulate
multiply-adds. Quantum layers (qml.qnn.TorchLayer) are reported separately
via a gate-count estimate (see src/models/quantum_attention.estimate_gate_count).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _conv_flops(m: nn.Conv2d, inp, out):
    out_t = out
    B, Cout, H, W = out_t.shape
    return B * Cout * H * W * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]


def _linear_flops(m: nn.Linear, inp, out):
    return out.numel() * m.in_features


def _ln_flops(m: nn.LayerNorm, inp, out):
    return out.numel() * 5  # rough constant


def _mha_flops(m: nn.MultiheadAttention, inp, out):
    # inp[0]: (B, N, D)
    x = inp[0]
    B, N, D = x.shape
    # Q,K,V projections + output proj + attention matmul + softmax-mul
    proj = 4 * B * N * D * D
    attn = 2 * B * m.num_heads * N * N * (D // m.num_heads)
    return proj + attn


HOOKS = {
    nn.Conv2d: _conv_flops,
    nn.Linear: _linear_flops,
    nn.LayerNorm: _ln_flops,
    nn.MultiheadAttention: _mha_flops,
}


def count_flops(model: nn.Module, sample: torch.Tensor) -> int:
    total = {"n": 0}
    handles = []

    def make_hook(fn):
        def hook(mod, inp, out):
            try:
                total["n"] += int(fn(mod, inp, out))
            except Exception:
                pass
        return hook

    for mod in model.modules():
        for cls, fn in HOOKS.items():
            if isinstance(mod, cls):
                handles.append(mod.register_forward_hook(make_hook(fn)))
                break

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample)
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return total["n"]
