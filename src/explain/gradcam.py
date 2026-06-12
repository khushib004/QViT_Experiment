"""Grad-CAM (Selvaraju et al., 2017) for the conv-based models (CNN, QCNN).

Produces a class-discriminative saliency map by weighting the target conv
layer's feature maps by the gradient of the predicted-class logit flowing
into them. Returns a normalized [0,1] heatmap upsampled to the input size.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or getattr(model, "gradcam_layer")
        self._activations = None
        self._gradients = None
        self._fwd = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """x: (1, 3, H, W). Returns (H, W) heatmap in [0,1]."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        grads = self._gradients          # (1, C, h, w)
        acts = self._activations         # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()
