"""Quantum Vision Transformer: ViT with QSA replacing classical MHA."""
from __future__ import annotations

from functools import partial

import torch.nn as nn

from .classical_vit import ViT
from .quantum_attention import HybridQuantumMultiHeadAttention


def qvit_tiny(
    num_classes: int = 2,
    n_qubits: int = 4,
    n_heads: int = 2,
    n_layers: int = 2,
    depth: int = 6,  # shallower than vit_tiny to keep sim cost manageable
    embed_dim: int = 96,
    device_name: str = "lightning.qubit",
    dropout: float = 0.0,
) -> ViT:
    attn_factory = partial(
        HybridQuantumMultiHeadAttention,
        n_heads=n_heads,
        n_qubits=n_qubits,
        n_layers=n_layers,
        device_name=device_name,
        dropout=dropout,
    )
    return ViT(
        embed_dim=embed_dim,
        depth=depth,
        heads=n_heads,
        num_classes=num_classes,
        attn_factory=attn_factory,
        drop=dropout,
    )
