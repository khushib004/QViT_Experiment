"""Unified saliency dispatcher — one function for any model type.

- conv-based models (CNN, QCNN) -> Grad-CAM
- transformer models (ViT, QViT) -> gradient-weighted attention rollout
  (class-specific, inspired by jacobgil/vit-explain and Chefer et al.)
  Falls back to vanilla rollout if backward pass fails.

So benchmark / visualization code can call one function and not care
about the model architecture.
"""
from __future__ import annotations

import numpy as np
import torch

from .attention_rollout import attention_rollout, gradient_attention_rollout
from .gradcam import GradCAM


def model_saliency(
    model,
    x: torch.Tensor,
    class_idx: int = None,
    prefer_grad: bool = True,
) -> np.ndarray:
    """x: (1,3,H,W). Returns (H,W) saliency in [0,1] for any supported model.

    For ViT/QViT, prefers gradient-weighted rollout (class-specific) by
    default. Set prefer_grad=False to force vanilla rollout.
    """
    if hasattr(model, "attention_maps"):
        if prefer_grad:
            try:
                return gradient_attention_rollout(model, x, class_idx=class_idx)
            except Exception:
                pass  # backward may fail on some VQC configs; fall back to vanilla
        return attention_rollout(model, x)

    if hasattr(model, "gradcam_layer"):
        cam = GradCAM(model)
        try:
            return cam(x, class_idx=class_idx)
        finally:
            cam.remove()

    raise TypeError(f"No saliency method available for {type(model).__name__}")
