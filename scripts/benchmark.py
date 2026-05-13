"""End-to-end benchmark: trains CNN, ViT, QCNN, QViT (+ optional KD) and
emits the three required plots plus a JSON results log.

Usage (Colab):
    !pip install pennylane pennylane-lightning[gpu] torch torchvision scikit-learn matplotlib
    !python scripts/benchmark.py --data_root /content/tyrenet --epochs 10 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import TyreNetDataset
from src.models import ClassicalCNN, QCNN, estimate_gate_count, qvit_tiny, vit_tiny
from src.training import history_to_dict, train_distilled, train_supervised
from src.utils import count_flops, plot_acc_vs_epochs, plot_acc_vs_params, plot_flops_vs_gates


def build_loaders(args):
    common = dict(
        root=args.data_root,
        n_qubits=args.n_qubits,
        patch_size=args.patch_size,
        reducer=args.reducer,
        return_quantum=False,  # all models consume the 224x224 classical view
    )
    train_ds = TyreNetDataset(split="train", **common)
    val_ds = TyreNetDataset(split="val", **common)
    test_ds = TyreNetDataset(split="test", **common)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--qvit_depth", type=int, default=4)
    ap.add_argument("--qvit_dim", type=int, default=96)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--reducer", default="conv", choices=["conv", "pca"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--qdevice", default="lightning.qubit")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--use_kd", action="store_true", help="Train QViT with KD from classical ViT teacher")
    ap.add_argument("--skip", nargs="*", default=[], choices=["cnn", "vit", "qcnn", "qvit"])
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(args)
    sample = next(iter(val_loader))["image_classical"][:1].to(args.device)

    records, histories = [], []

    def run(name, model, hist_fn):
        h = hist_fn()
        records.append({
            "model_name": name,
            "n_params": h.n_params,
            "best_val_acc": max(m.val_acc for m in h.history),
            "flops": count_flops(model.to(args.device), sample),
            "quantum_gates": gate_counts.get(name, 0),
        })
        histories.append(history_to_dict(h))

    gate_counts = {}

    # --- Classical CNN ---
    if "cnn" not in args.skip:
        cnn = ClassicalCNN()
        run("CNN", cnn, lambda: train_supervised(
            cnn, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
            device=args.device, model_name="CNN",
        ))

    # --- Classical ViT (teacher for KD) ---
    teacher = None
    if "vit" not in args.skip:
        teacher = vit_tiny(num_classes=2)
        run("ViT-Tiny", teacher, lambda: train_supervised(
            teacher, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
            device=args.device, model_name="ViT-Tiny",
        ))

    # --- QCNN ---
    if "qcnn" not in args.skip:
        qcnn = QCNN(n_qubits=args.n_qubits, n_layers=args.n_layers, device_name=args.qdevice)
        gate_counts["QCNN"] = estimate_gate_count(
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            n_tokens=(28 // 2) ** 2, n_heads=1,
        )["total_gates"] // 3  # not Q/K/V, just one circuit per patch
        run("QCNN", qcnn, lambda: train_supervised(
            qcnn, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
            device=args.device, model_name="QCNN",
        ))

    # --- QViT (optionally distilled from ViT) ---
    if "qvit" not in args.skip:
        qvit = qvit_tiny(
            num_classes=2,
            n_qubits=args.n_qubits,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            depth=args.qvit_depth,
            embed_dim=args.qvit_dim,
            device_name=args.qdevice,
        )
        n_tokens = (224 // 16) ** 2 + 1  # patches + CLS
        gate_counts["QViT"] = estimate_gate_count(
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            n_tokens=n_tokens, n_heads=args.n_heads,
        )["total_gates"] * args.qvit_depth

        if args.use_kd and teacher is not None:
            run("QViT (KD)", qvit, lambda: train_distilled(
                qvit, teacher, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                device=args.device, model_name="QViT-KD",
            ))
        else:
            run("QViT", qvit, lambda: train_supervised(
                qvit, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                device=args.device, model_name="QViT",
            ))

    plot_acc_vs_params(records, str(out_dir / "plots" / "acc_vs_params.png"))
    plot_acc_vs_epochs(histories, str(out_dir / "plots" / "acc_vs_epochs.png"))
    plot_flops_vs_gates(records, str(out_dir / "plots" / "flops_vs_gates.png"))

    with open(out_dir / "logs" / "benchmark.json", "w") as f:
        json.dump({"records": records, "histories": histories, "args": vars(args)}, f, indent=2)

    print("\n== Summary ==")
    for r in records:
        print(f"{r['model_name']:>10s}  params={r['n_params']:>9d}  "
              f"best_val={r['best_val_acc']:.3f}  flops={r['flops']:.2e}  "
              f"qgates={r['quantum_gates']:>8d}")


if __name__ == "__main__":
    main()
