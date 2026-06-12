"""Heatmap overlay + multi-model comparison rendering.

Turns a [0,1] saliency map into a JET overlay on the original (de-normalized)
tyre image, and arranges per-model saliencies into a single comparison figure
answering: "where is each architecture looking, and does it find the defect?"
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) uint8 RGB."""
    x = img.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _get_cmap(name: str):
    """Version-robust colormap lookup (matplotlib >=3.9 deprecated cm.get_cmap)."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _jet(gray: np.ndarray) -> np.ndarray:
    """JET colormap, returns (H,W,3) uint8."""
    return (_get_cmap("jet")(gray)[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(img: torch.Tensor, sal: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = denormalize(img).astype(np.float32)
    heat = _jet(np.clip(sal, 0, 1)).astype(np.float32)
    out = (1 - alpha) * base + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def compare_models_figure(
    img: torch.Tensor,
    saliencies: Dict[str, np.ndarray],
    out_path: Optional[str] = None,
    title: str = "",
    defect_mask: Optional[np.ndarray] = None,
):
    """Render [original | model_1 overlay | model_2 overlay | ...].

    If `defect_mask` (H,W in {0,1}) is given, also annotates each panel with a
    'defect focus' score = mean saliency inside the mask.
    """
    import matplotlib.pyplot as plt

    n = 1 + len(saliencies)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]

    axes[0].imshow(denormalize(img))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")
    if defect_mask is not None:
        axes[0].contour(defect_mask, levels=[0.5], colors="lime", linewidths=1.5)

    for ax, (name, sal) in zip(axes[1:], saliencies.items()):
        ax.imshow(overlay_heatmap(img, sal))
        sub = name
        if defect_mask is not None and defect_mask.sum() > 0:
            focus = float((sal * defect_mask).sum() / (defect_mask.sum() + 1e-8))
            sub = f"{name}\ndefect-focus={focus:.2f}"
        ax.set_title(sub, fontsize=11)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig


def defect_focus_score(sal: np.ndarray, defect_mask: np.ndarray) -> float:
    """Fraction of saliency mass that lands on the annotated defect region.

    A higher score means the model 'looks at' the real defect — a concrete,
    quantitative way to compare model attention quality beyond raw accuracy.
    """
    if defect_mask.sum() == 0:
        return float("nan")
    inside = float((sal * defect_mask).sum())
    total = float(sal.sum()) + 1e-8
    return inside / total
