"""Unified saliency dispatcher: pick the right method per model type.

- conv-based models (CNN, QCNN) -> Grad-CAM
- transformer models (ViT, QViT) -> attention rollout

So benchmark/visualization code can call one function for any model.
"""
from __future__ import annotations

import numpy as np
import torch

from .attention_rollout import attention_rollout
from .gradcam import GradCAM


def model_saliency(model, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
    """x: (1,3,H,W). Returns (H,W) saliency in [0,1] for any supported model."""
    if hasattr(model, "attention_maps"):
        return attention_rollout(model, x)
    if hasattr(model, "gradcam_layer"):
        cam = GradCAM(model)
        try:
            return cam(x, class_idx=class_idx)
        finally:
            cam.remove()
    raise TypeError(f"No saliency method available for {type(model).__name__}")
