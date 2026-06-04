"""
Hybrid Quantum Self-Attention (QSA) for Quantum Vision Transformers.

Theory
------
Classical Multi-Head Self-Attention learns three D x D projection matrices
(W_Q, W_K, W_V) per head -> 3*D^2 parameters, i.e. O(D^2). Quantum Self-
Attention replaces each projection with a per-token Variational Quantum
Circuit (VQC) whose only trainable tensor scales as O(n_qubits) (Cherrat
et al. 2024; Boucher et al. 2025).

Circuit (per token)
-------------------
  1. State prep:  AngleEmbedding(R_Y) of the n_qubits reduced features.
  2. Ansatz:      BasicEntanglerLayers OR StronglyEntanglingLayers, with
                  optional *data re-uploading* (Perez-Salinas 2020) that
                  re-embeds the input before every variational layer to
                  boost expressivity at fixed qubit count.
  3. Readout:     <Z_i> on each wire -> n_qubits-dim per-token vector.

Explainability
--------------
Every attention module caches its last attention matrix in `.last_attn`
(shape (B, heads, N, N)). This is consumed by attention-rollout to render
"where the model looks" heatmaps over the original tyre image.
"""
from __future__ import annotations

import math
from typing import Optional

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# QNode construction
# --------------------------------------------------------------------------- #
def _build_qnode(
    n_qubits: int,
    n_layers: int,
    device_name: str = "lightning.qubit",
    ansatz: str = "basic",
    reupload: bool = True,
):
    """Build a TorchLayer-compatible QNode.

    ansatz:    'basic' -> BasicEntanglerLayers (1 param/qubit/layer)
               'strong' -> StronglyEntanglingLayers (3 params/qubit/layer)
    reupload:  re-embed the input before each variational layer.
    """
    dev = qml.device(device_name, wires=n_qubits)

    if ansatz == "strong":
        weight_shape = (n_layers, n_qubits, 3)
        ansatz_fn = qml.StronglyEntanglingLayers
    else:
        weight_shape = (n_layers, n_qubits)
        ansatz_fn = qml.BasicEntanglerLayers

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        if reupload:
            # Interleave embedding and one variational layer at a time.
            for layer in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                ansatz_fn(weights[layer : layer + 1], wires=range(n_qubits))
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            ansatz_fn(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit, weight_shape


class QuantumLinear(nn.Module):
    """Token-wise quantum 'projection': one VQC applied to every token.

    Input/Output: (B, N, n_qubits). The single registered weight tensor gives
    the O(n_qubits) scaling at the heart of the QSA efficiency claim.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        device_name: str = "lightning.qubit",
        ansatz: str = "basic",
        reupload: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        qnode, weight_shape = _build_qnode(n_qubits, n_layers, device_name, ansatz, reupload)
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": weight_shape})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.n_qubits, f"expected last dim {self.n_qubits}, got {D}"
        out = self.qlayer(x.reshape(B * N, D))
        return out.reshape(B, N, D)


class QuantumSelfAttentionHead(nn.Module):
    """A single quantum self-attention head.

    Classical projections compress embed_dim -> n_qubits (kept small so the
    register is simulable). Q, K, V each pass through their own VQC; attention
    weights are the classical scaled dot-product softmax of the quantum Q, K.
    The per-head attention matrix is exposed for visualization.
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = "lightning.qubit",
        ansatz: str = "basic",
        reupload: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.scale = 1.0 / math.sqrt(n_qubits)

        # Small bottleneck MLPs give the VQC cleaner inputs than a bare linear.
        def proj():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, n_qubits),
            )

        self.q_proj, self.k_proj, self.v_proj = proj(), proj(), proj()
        qkw = dict(n_qubits=n_qubits, n_layers=n_layers, device_name=device_name,
                   ansatz=ansatz, reupload=reupload)
        self.q_vqc = QuantumLinear(**qkw)
        self.k_vqc = QuantumLinear(**qkw)
        self.v_vqc = QuantumLinear(**qkw)
        self.attn_drop = nn.Dropout(dropout)
        self.last_attn: Optional[torch.Tensor] = None  # (B, N, N)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tanh*pi keeps AngleEmbedding inputs in the maximally expressive [-pi, pi].
        q = self.q_vqc(torch.tanh(self.q_proj(x)) * math.pi)
        k = self.k_vqc(torch.tanh(self.k_proj(x)) * math.pi)
        v = self.v_vqc(torch.tanh(self.v_proj(x)) * math.pi)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        self.last_attn = attn.detach()
        return torch.matmul(self.attn_drop(attn), v)  # (B, N, n_qubits)


class HybridQuantumMultiHeadAttention(nn.Module):
    """Multi-head QSA: concatenate heads, project back to embed_dim.

    Caches `.last_attn` as (B, n_heads, N, N) for attention rollout.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = "lightning.qubit",
        ansatz: str = "basic",
        reupload: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                QuantumSelfAttentionHead(
                    embed_dim, n_qubits, n_layers, device_name, ansatz, reupload, dropout
                )
                for _ in range(n_heads)
            ]
        )
        self.out_proj = nn.Linear(n_heads * n_qubits, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outs = [h(x, mask=mask) for h in self.heads]
        self.last_attn = torch.stack([h.last_attn for h in self.heads], dim=1)  # (B, H, N, N)
        return self.proj_drop(self.out_proj(torch.cat(outs, dim=-1)))


def estimate_gate_count(
    n_qubits: int,
    n_layers: int,
    n_tokens: int,
    n_heads: int,
    ansatz: str = "basic",
    reupload: bool = True,
) -> dict:
    """Analytical per-batch-element gate budget for one attention forward pass."""
    embeds = n_layers if reupload else 1
    embed_1q = embeds * n_qubits  # R_Y rotations
    if ansatz == "strong":
        ansatz_1q = n_layers * n_qubits * 3
        ansatz_2q = n_layers * n_qubits  # ring of CNOTs
    else:
        ansatz_1q = n_layers * n_qubits
        ansatz_2q = n_layers * n_qubits
    per_circuit_1q = embed_1q + ansatz_1q
    per_circuit_2q = ansatz_2q
    circuits = 3 * n_heads * n_tokens  # Q, K, V per head per token
    return {
        "single_qubit_gates": per_circuit_1q * circuits,
        "two_qubit_gates": per_circuit_2q * circuits,
        "total_gates": (per_circuit_1q + per_circuit_2q) * circuits,
    }
