"""Quantum-native visualizations that make the 'quantum' tangible.

- draw_circuit:        ASCII + matplotlib render of the VQC used by QSA/QCNN.
- bloch_vectors:       per-qubit Bloch sphere coordinates (<X>,<Y>,<Z>) for a
                       given input, so you can *see* state preparation + the
                       effect of the entangling ansatz.
- measurement_hist:    sampled bitstring distribution from the circuit.
- expressivity_curve:  how output variance grows with ansatz depth (a proxy
                       for the circuit's expressive power vs. barren plateaus).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pennylane as qml
import torch


def draw_circuit(n_qubits: int = 4, n_layers: int = 2, device_name: str = "default.qubit",
                 ansatz: str = "basic", reupload: bool = True) -> str:
    """Return an ASCII drawing of the QSA circuit."""
    from src.models.quantum_attention import _build_qnode

    qnode, wshape = _build_qnode(n_qubits, n_layers, device_name, ansatz, reupload)
    dummy_in = torch.zeros(n_qubits)
    dummy_w = torch.zeros(*wshape)
    return qml.draw(qnode)(dummy_in, dummy_w)


def bloch_vectors(inputs: np.ndarray, n_qubits: int = 4, n_layers: int = 2,
                  weights: np.ndarray = None, device_name: str = "default.qubit") -> np.ndarray:
    """Return (n_qubits, 3) array of (<X>,<Y>,<Z>) per qubit after the circuit."""
    dev = qml.device(device_name, wires=n_qubits)

    if weights is None:
        weights = np.zeros((n_layers, n_qubits))

    @qml.qnode(dev)
    def circuit(x, w):
        for layer in range(n_layers):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(w[layer:layer + 1], wires=range(n_qubits))
        return [qml.expval(o(w_)) for w_ in range(n_qubits) for o in (qml.PauliX, qml.PauliY, qml.PauliZ)]

    vals = np.array(circuit(inputs, weights)).reshape(n_qubits, 3)
    return vals


def plot_bloch(vectors: np.ndarray, title: str = "Qubit states after the VQC", out_path: str = None):
    """3-D Bloch sphere scatter of per-qubit (<X>,<Y>,<Z>) vectors."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    # wireframe sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="lightgray", linewidth=0.4)
    for i, (x, y, z) in enumerate(vectors):
        ax.quiver(0, 0, 0, x, y, z, length=1.0, normalize=False, linewidth=2)
        ax.text(x, y, z, f"q{i}", fontsize=9)
    ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig


def expressivity_curve(n_qubits: int = 4, max_layers: int = 6, n_samples: int = 64,
                       device_name: str = "default.qubit", out_path: str = None):
    """Plot output-variance vs. ansatz depth (expressivity proxy)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    depths = list(range(1, max_layers + 1))
    variances = []
    for L in depths:
        dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x, w):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(w, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        outs = []
        for _ in range(n_samples):
            x = rng.uniform(-np.pi, np.pi, size=n_qubits)
            w = rng.uniform(-np.pi, np.pi, size=(L, n_qubits))
            outs.append(float(circuit(x, w)))
        variances.append(np.var(outs))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(depths, variances, marker="o")
    ax.set_xlabel("Ansatz depth (n_layers)")
    ax.set_ylabel("Var[⟨Z₀⟩] over random inputs")
    ax.set_title("Circuit expressivity vs. depth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig
