"""Evaluation metrics + the plots that turn results into evidence.

- collect_predictions:  logits/probs/labels over a loader.
- classification_report: accuracy, precision, recall, F1, AUC.
- plot_confusion / plot_roc_pr: standard diagnostic curves.
- plot_pareto_frontier:  accuracy vs. #params with the efficiency frontier
                         highlighted — the single most important figure for
                         the parameter-efficiency hypothesis.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs (N,C), preds (N,), labels (N,))."""
    model.eval().to(device)
    probs, labels = [], []
    for batch in loader:
        x = batch["image_classical"].to(device)
        p = F.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(batch["label"].numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return probs, probs.argmax(1), labels


def classification_report(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(len(labels), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs[:, 1])) if len(set(labels)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def plot_confusion(preds: np.ndarray, labels: np.ndarray, title: str, out_path: str = None):
    import matplotlib.pyplot as plt

    cm = np.zeros((2, 2), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["good", "defective"]); ax.set_yticklabels(["good", "defective"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig


def plot_roc_pr(results: List[Dict], out_path: str = None):
    """results: [{'name','probs','labels'}, ...] -> overlaid ROC + PR curves."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for r in results:
        y, s = r["labels"], r["probs"][:, 1]
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax1.plot(fpr, tpr, label=r["name"])
        prec, rec, _ = precision_recall_curve(y, s)
        ax2.plot(rec, prec, label=r["name"])
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision–Recall"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig


def plot_pareto_frontier(records: List[Dict], out_path: str = None):
    """records: [{'model_name','n_params','best_val_acc', 'is_quantum'(opt)}].

    Draws accuracy vs. log-params, connects the Pareto-optimal points, and
    annotates each model. Quantum models are highlighted as stars.
    """
    import matplotlib.pyplot as plt

    pts = sorted(records, key=lambda r: r["n_params"])
    fig, ax = plt.subplots(figsize=(7, 5))

    # Pareto frontier: points not dominated (fewer params AND >= acc).
    frontier = []
    best_acc = -1
    for r in pts:
        if r["best_val_acc"] > best_acc:
            frontier.append(r)
            best_acc = r["best_val_acc"]
    fx = [r["n_params"] for r in frontier]
    fy = [r["best_val_acc"] for r in frontier]
    ax.plot(fx, fy, "--", color="gray", alpha=0.6, zorder=1, label="efficiency frontier")

    for r in records:
        q = r.get("is_quantum", "Q" in r["model_name"])
        ax.scatter(r["n_params"], r["best_val_acc"],
                   s=240 if q else 120,
                   marker="*" if q else "o",
                   c="#e63946" if q else "#1d3557",
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(r["model_name"], (r["n_params"], r["best_val_acc"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Parameter efficiency: accuracy vs. model size\n(★ = quantum hybrid)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig
