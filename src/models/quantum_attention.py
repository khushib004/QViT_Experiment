"""
Hybrid Quantum Self-Attention (QSA) head for Quantum Vision Transformers.

Theoretical basis:
    - Cherrat et al. (2024): Quantum Vision Transformers, showing O(n) parameter
      scaling of QSA compared to O(n^2) of classical self-attention.
    - Boucher et al. (2025): on the inductive bias of variational quantum
      attention for global feature extraction.

Design:
    Each attention head is realized by a Variational Quantum Circuit (VQC):
        1. AngleEmbedding maps a token's reduced feature vector (size = n_qubits)
           to single-qubit rotations (state preparation).
        2. BasicEntanglerLayers provides a hardware-efficient ansatz that
           entangles qubits across CNOT rings (variational ansatz).
        3. Pauli-Z expectations on each wire produce the per-token output
           of size n_qubits (the "value-attended" representation).

    Multi-head behaviour is achieved by replicating the circuit with different
    trainable parameters and concatenating outputs along the feature axis.

    Classical pre-projections (q_proj, k_proj, v_proj) compress the embedding
    dim D to n_qubits so that the quantum register stays small (4-8 qubits) —
    critical for tractable simulation on `lightning.gpu`.
"""
from __future__ import annotations

import math
from typing import Optional

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_qnode(n_qubits: int, n_layers: int, device_name: str = "lightning.qubit"):
    """Construct a PennyLane QNode wired to a TorchLayer-friendly interface."""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        # State preparation: angle-embed reduced features as RY rotations.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # Variational ansatz: BasicEntanglerLayers (RX + ring of CNOTs).
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        # Measure each wire to get an n_qubits-dimensional output.
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


class QuantumLinear(nn.Module):
    """A token-wise quantum 'projection' realized by a TorchLayer over a VQC.

    Applies the same VQC independently to every token in a (B, N, n_qubits)
    tensor. The TorchLayer is registered with a single weight tensor of shape
    (n_layers, n_qubits), giving the O(n) parameter scaling claimed by QSA.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2, device_name: str = "lightning.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        qnode = _build_qnode(n_qubits, n_layers, device_name)
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, n_qubits) -> flatten tokens, apply VQC, restore shape.
        B, N, D = x.shape
        assert D == self.n_qubits, f"Expected last dim={self.n_qubits}, got {D}"
        flat = x.reshape(B * N, D)
        out = self.qlayer(flat)
        return out.reshape(B, N, D)


class QuantumSelfAttentionHead(nn.Module):
    """A single QSA head.

    The classical projections (q/k/v) compress embed_dim -> n_qubits so the
    quantum register remains small. Q and K are mapped through independent
    VQCs; the resulting Q,K are used to compute classical scaled dot-product
    attention weights. V is also processed by a quantum 'value' layer, giving
    a fully quantum-feature pipeline while keeping the softmax classical
    (the only practical option on NISQ hardware).
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = "lightning.qubit",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.scale = 1.0 / math.sqrt(n_qubits)

        self.q_proj = nn.Linear(embed_dim, n_qubits, bias=False)
        self.k_proj = nn.Linear(embed_dim, n_qubits, bias=False)
        self.v_proj = nn.Linear(embed_dim, n_qubits, bias=False)

        self.q_vqc = QuantumLinear(n_qubits, n_layers, device_name)
        self.k_vqc = QuantumLinear(n_qubits, n_layers, device_name)
        self.v_vqc = QuantumLinear(n_qubits, n_layers, device_name)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, embed_dim)
        q = self.q_vqc(torch.tanh(self.q_proj(x)))  # tanh keeps inputs in [-1,1] for AngleEmbedding
        k = self.k_vqc(torch.tanh(self.k_proj(x)))
        v = self.v_vqc(torch.tanh(self.v_proj(x)))

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_drop(F.softmax(attn_logits, dim=-1))
        return torch.matmul(attn, v)  # (B, N, n_qubits)


class HybridQuantumMultiHeadAttention(nn.Module):
    """Multi-head wrapper that concatenates QSA heads and projects back to embed_dim.

    Trainable-parameter budget is approximately:
        n_heads * (3 * embed_dim * n_qubits + 3 * n_layers * n_qubits)
                + (n_heads * n_qubits) * embed_dim
    which is linear in n_qubits inside the quantum block — the O(n) scaling
    discussed in Cherrat et al. (2024).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = "lightning.qubit",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                QuantumSelfAttentionHead(
                    embed_dim=embed_dim,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    device_name=device_name,
                    dropout=dropout,
                )
                for _ in range(n_heads)
            ]
        )
        self.out_proj = nn.Linear(n_heads * n_qubits, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        head_outs = [h(x, mask=mask) for h in self.heads]
        concat = torch.cat(head_outs, dim=-1)  # (B, N, n_heads * n_qubits)
        return self.proj_drop(self.out_proj(concat))


def estimate_gate_count(n_qubits: int, n_layers: int, n_tokens: int, n_heads: int) -> dict:
    """Rough quantum-gate accounting for one forward pass (per batch element).

    AngleEmbedding contributes n_qubits single-qubit rotations.
    BasicEntanglerLayers contributes n_layers * (n_qubits RX + n_qubits CNOTs).
    Applied per-token, per-head, for Q, K, V projections (3x).
    """
    per_circuit_1q = n_qubits + n_layers * n_qubits
    per_circuit_2q = n_layers * n_qubits  # ring CNOTs
    circuits = 3 * n_heads * n_tokens  # Q, K, V
    return {
        "single_qubit_gates": per_circuit_1q * circuits,
        "two_qubit_gates": per_circuit_2q * circuits,
        "total_gates": (per_circuit_1q + per_circuit_2q) * circuits,
    }
