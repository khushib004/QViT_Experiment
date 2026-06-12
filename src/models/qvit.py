"""Quantum Vision Transformer: ViT with Quantum Self-Attention blocks.

Inherits attention-map capture and `.embed()` from the ViT base, so all the
explainability tooling (attention rollout, t-SNE) works unchanged.
"""
from __future__ import annotations

from functools import partial

from .classical_vit import ViT
from .quantum_attention import HybridQuantumMultiHeadAttention


def qvit_tiny(
    num_classes: int = 2,
    n_qubits: int = 4,
    n_heads: int = 2,
    n_layers: int = 2,
    depth: int = 4,
    embed_dim: int = 96,
    patch_size: int = 16,
    ansatz: str = "basic",       # 'basic' | 'strong'
    reupload: bool = True,
    device_name: str = "lightning.qubit",
    dropout: float = 0.0,
) -> ViT:
    attn_factory = partial(
        HybridQuantumMultiHeadAttention,
        n_heads=n_heads,
        n_qubits=n_qubits,
        n_layers=n_layers,
        device_name=device_name,
        ansatz=ansatz,
        reupload=reupload,
        dropout=dropout,
    )
    return ViT(
        img_size=224,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        heads=n_heads,
        num_classes=num_classes,
        attn_factory=attn_factory,
        drop=dropout,
    )
