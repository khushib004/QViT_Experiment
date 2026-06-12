"""Plotting helpers for the benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_acc_vs_params(records: List[Dict], out: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in records:
        ax.scatter(r["n_params"], r["best_val_acc"], s=80, label=r["model_name"])
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log)")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Accuracy vs. parameter budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_acc_vs_epochs(histories: List[Dict], out: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for h in histories:
        epochs = [m["epoch"] for m in h["history"]]
        accs = [m["val_acc"] for m in h["history"]]
        ax.plot(epochs, accs, marker="o", label=h["model_name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs. training epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_flops_vs_gates(records: List[Dict], out: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in records:
        ax.scatter(r.get("flops", 0), r.get("quantum_gates", 0), s=80, label=r["model_name"])
    ax.set_xlabel("Classical FLOPs / forward (log)")
    ax.set_ylabel("Quantum gate count / forward")
    ax.set_xscale("log")
    ax.set_title("Compute footprint: classical FLOPs vs. quantum gates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
