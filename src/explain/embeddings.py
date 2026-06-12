"""Learned-feature visualization: extract embeddings and project to 2-D.

Shows whether a model carves a clean linear boundary between 'good' and
'defective' in its representation space — strong qualitative evidence of
representation quality that complements raw accuracy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(model, loader: DataLoader, device: str, max_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embeddings (N,D), labels (N,)) using the model's `.embed()`."""
    model.eval().to(device)
    embs, labels = [], []
    seen = 0
    for batch in loader:
        x = batch["image_classical"].to(device)
        y = batch["label"]
        e = model.embed(x).cpu().numpy()
        embs.append(e)
        labels.append(y.numpy())
        seen += x.size(0)
        if seen >= max_samples:
            break
    return np.concatenate(embs)[:max_samples], np.concatenate(labels)[:max_samples]


def project_2d(emb: np.ndarray, method: str = "tsne", seed: int = 0) -> np.ndarray:
    """Project (N,D) embeddings to (N,2) via t-SNE (default) or PCA."""
    if method == "pca" or emb.shape[0] < 10:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(emb)
    from sklearn.manifold import TSNE
    perp = min(30, max(5, emb.shape[0] // 4))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(emb)


def plot_embedding(emb2d: np.ndarray, labels: np.ndarray, title: str, out_path: str = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for cls, name, color in [(0, "good", "#2a9d8f"), (1, "defective", "#e76f51")]:
        m = labels == cls
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=18, alpha=0.7, label=name, c=color, edgecolors="none")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig
