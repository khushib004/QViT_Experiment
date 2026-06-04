"""Hybrid Quantum CNN ('Quanvolutional' architecture, Henderson et al., 2019).

A Quanvolutional layer slides a small VQC over local 2x2 image patches and
emits one feature map per measured qubit:
  - 2x2 stride-2 patches feed n_qubits>=4
  - AngleEmbedding(R_Y) + BasicEntanglerLayers kernel
  - per-wire <Z> form the output channels

The quantum feature maps are *themselves* directly visualizable (one heatmap
per qubit), and the classifier's last conv is a Grad-CAM target.
"""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


def _quanv_qnode(n_qubits: int, n_layers: int, device_name: str = "lightning.qubit"):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


class QuanvLayer(nn.Module):
    """A 'Quanvolutional' layer with 2x2 stride-2 kernel."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device_name: str = "lightning.qubit"):
        super().__init__()
        assert n_qubits >= 4, "2x2 patch needs at least 4 qubits"
        self.n_qubits = n_qubits
        qnode = _quanv_qnode(n_qubits, n_layers, device_name)
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})
        self.last_quantum_maps = None  # (B, n_qubits, H', W') cached for viz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = torch.tanh(x) * 3.14159265
        patches = F.unfold(x, kernel_size=2, stride=2)  # (B, 4, L)
        L = patches.shape[-1]
        if patches.shape[1] < self.n_qubits:
            pad = torch.zeros(B, self.n_qubits - patches.shape[1], L, device=x.device)
            patches = torch.cat([patches, pad], dim=1)
        flat = patches.permute(0, 2, 1).reshape(B * L, self.n_qubits)
        out = self.qlayer(flat).reshape(B, L, self.n_qubits).permute(0, 2, 1)
        maps = out.reshape(B, self.n_qubits, H // 2, W // 2)
        self.last_quantum_maps = maps.detach()
        return maps


class QCNN(nn.Module):
    """Quanvolutional front-end + small classical CNN classifier."""

    def __init__(
        self,
        num_classes: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = "lightning.qubit",
        input_size: int = 28,
    ):
        super().__init__()
        self.input_size = input_size
        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.to_gray.weight, 1.0 / 3.0)
        for p in self.to_gray.parameters():
            p.requires_grad_(False)

        self.quanv = QuanvLayer(n_qubits=n_qubits, n_layers=n_layers, device_name=device_name)
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_qubits, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        # Grad-CAM target.
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)
        self._feature_maps = None

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.conv2

    def forward_features(self, x):
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = self.to_gray(x)
        x = self.quanv(x)
        h = self.conv2(self.conv1(x))
        self._feature_maps = h
        return h

    def forward(self, x):
        h = self.forward_features(x)
        return self.classifier(self.pool(h).flatten(1))

    @torch.no_grad()
    def embed(self, x):
        return self.pool(self.forward_features(x)).flatten(1)
