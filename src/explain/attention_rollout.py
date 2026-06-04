"""Attention rollout (Abnar & Zuidema, 2020) for ViT and QViT.

Recursively multiplies the per-layer attention matrices (after adding the
residual connection as identity and re-normalizing) to estimate how much
each input patch contributes to the final CLS token. The CLS->patches row
of the rolled-out matrix becomes a saliency map over the patch grid.

Works for both classical ViT (ClassicalMHA.last_attn) and QViT
(HybridQuantumMultiHeadAttention.last_attn) because both expose attention
tensors of shape (B, heads, N, N) via `model.attention_maps()`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def attention_rollout(
    model,
    x: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """x: (1, 3, H, W). Returns (H, W) saliency in [0,1].

    head_fusion:   'mean' | 'max' | 'min' across heads.
    discard_ratio: drop the lowest-attention edges to sharpen the map.
    """
    model.eval()
    _ = model(x)  # populate per-block attention caches
    mats = model.attention_maps()  # list of (B, heads, N, N)
    if not mats:
        raise RuntimeError("Model exposed no attention maps; is it a ViT/QViT?")

    N = mats[0].shape[-1]
    result = torch.eye(N, device=mats[0].device)
    for attn in mats:
        a = attn[0]  # (heads, N, N)
        if head_fusion == "max":
            fused = a.max(dim=0).values
        elif head_fusion == "min":
            fused = a.min(dim=0).values
        else:
            fused = a.mean(dim=0)

        if discard_ratio > 0:
            flat = fused.view(-1)
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, largest=False)
                flat[idx] = 0
                fused = flat.view_as(fused)

        # Add identity (residual) and renormalize rows.
        fused = fused + torch.eye(N, device=fused.device)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        result = fused @ result

    # CLS token attends to the patch tokens (drop the CLS->CLS entry).
    cls_to_patches = result[0, 1:]          # (N-1,)
    grid = int(round((cls_to_patches.numel()) ** 0.5))
    sal = cls_to_patches.reshape(1, 1, grid, grid)
    sal = F.interpolate(sal, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.cpu().numpy()
