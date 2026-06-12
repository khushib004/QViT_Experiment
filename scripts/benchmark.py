"""End-to-end benchmark + evidence suite for the QViT TyreNet study.

Trains CNN, ViT, QCNN, QViT (+ optional KD) and emits:
  Core efficiency plots
    - acc_vs_params.png, acc_vs_epochs.png, flops_vs_gates.png
    - pareto_frontier.png            (the parameter-efficiency proof)
  Diagnostic plots (per model + overlaid)
    - confusion_<model>.png, roc_pr.png
  Explainability (the 'where does each model look?' evidence)
    - saliency_compare_<i>.png       (original | CNN | ViT | QCNN | QViT)
    - embeddings_<model>.png         (t-SNE of learned features)
  Machine-readable
    - results/logs/benchmark.json

Usage (Colab):
    !pip install pennylane "pennylane-lightning[gpu]" torch torchvision scikit-learn matplotlib
    !python scripts/benchmark.py --data_root data/tyrenet --epochs 20 --qdevice lightning.gpu --use_kd
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
from src.explain import (
    compare_models_figure,
    extract_embeddings,
    model_saliency,
    plot_embedding,
    project_2d,
)
from src.models import ClassicalCNN, QCNN, estimate_gate_count, qvit_tiny, vit_tiny
from src.training import history_to_dict, train_distilled, train_supervised
from src.utils import (
    classification_report,
    collect_predictions,
    count_flops,
    plot_acc_vs_epochs,
    plot_acc_vs_params,
    plot_confusion,
    plot_flops_vs_gates,
    plot_pareto_frontier,
    plot_roc_pr,
)


def build_loaders(args):
    common = dict(root=args.data_root, n_qubits=args.n_qubits, patch_size=args.patch_size,
                  reducer=args.reducer, return_quantum=False)
    train_ds = TyreNetDataset(split="train", **common)
    val_ds = TyreNetDataset(split="val", **common)
    test_ds = TyreNetDataset(split="test", **common)
    mk = lambda ds, s: DataLoader(ds, batch_size=args.batch_size, shuffle=s,
                                  num_workers=args.workers, pin_memory=True)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), test_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--qvit_depth", type=int, default=4)
    ap.add_argument("--qvit_dim", type=int, default=96)
    ap.add_argument("--qvit_patch", type=int, default=16)
    ap.add_argument("--ansatz", default="basic", choices=["basic", "strong"])
    ap.add_argument("--no_reupload", action="store_true")
    ap.add_argument("--quantum_warmup", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--reducer", default="conv", choices=["conv", "pca"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--qdevice", default="lightning.qubit")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--use_kd", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=["cnn", "vit", "qcnn", "qvit"])
    ap.add_argument("--n_saliency", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.results_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, test_ds = build_loaders(args)
    sample = next(iter(val_loader))["image_classical"][:1].to(args.device)

    records, histories, trained = [], [], {}
    gate_counts = {}

    def finalize(name, model, hist, qgates=0):
        probs, preds, labels = collect_predictions(model, test_loader, args.device)
        rep = classification_report(probs, preds, labels)
        plot_confusion(preds, labels, f"{name} (test)", str(out / "plots" / f"confusion_{name}.png"))
        records.append({
            "model_name": name,
            "n_params": hist.n_params,
            "best_val_acc": hist.best_val_acc,
            "test_acc": rep["accuracy"],
            "test_f1": rep["f1"],
            "test_auc": rep["auc"],
            "flops": count_flops(model.to(args.device), sample),
            "quantum_gates": qgates,
            "is_quantum": name.startswith("Q"),
        })
        histories.append(history_to_dict(hist))
        trained[name] = {"model": model, "probs": probs, "labels": labels}

    # --- CNN ---
    if "cnn" not in args.skip:
        cnn = ClassicalCNN()
        h = train_supervised(cnn, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                             device=args.device, model_name="CNN")
        finalize("CNN", cnn, h)

    # --- ViT (teacher) ---
    teacher = None
    if "vit" not in args.skip:
        teacher = vit_tiny(num_classes=2)
        h = train_supervised(teacher, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                             device=args.device, model_name="ViT")
        finalize("ViT", teacher, h)

    # --- QCNN ---
    if "qcnn" not in args.skip:
        qcnn = QCNN(n_qubits=args.n_qubits, n_layers=args.n_layers, device_name=args.qdevice)
        gate_counts["QCNN"] = estimate_gate_count(args.n_qubits, args.n_layers, (28 // 2) ** 2, 1,
                                                  ansatz="basic", reupload=False)["total_gates"] // 3
        h = train_supervised(qcnn, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                             quantum_warmup=args.quantum_warmup, device=args.device, model_name="QCNN")
        finalize("QCNN", qcnn, h, gate_counts["QCNN"])

    # --- QViT (optionally distilled) ---
    if "qvit" not in args.skip:
        qvit = qvit_tiny(num_classes=2, n_qubits=args.n_qubits, n_heads=args.n_heads,
                         n_layers=args.n_layers, depth=args.qvit_depth, embed_dim=args.qvit_dim,
                         patch_size=args.qvit_patch, ansatz=args.ansatz,
                         reupload=not args.no_reupload, device_name=args.qdevice)
        n_tokens = (224 // args.qvit_patch) ** 2 + 1
        gate_counts["QViT"] = estimate_gate_count(args.n_qubits, args.n_layers, n_tokens, args.n_heads,
                                                  ansatz=args.ansatz, reupload=not args.no_reupload
                                                  )["total_gates"] * args.qvit_depth
        if args.use_kd and teacher is not None:
            h = train_distilled(qvit, teacher, train_loader, val_loader, epochs=args.epochs,
                                lr=args.lr, device=args.device, model_name="QViT-KD")
            finalize("QViT", qvit, h, gate_counts["QViT"])
        else:
            h = train_supervised(qvit, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                                 quantum_warmup=args.quantum_warmup, device=args.device, model_name="QViT")
            finalize("QViT", qvit, h, gate_counts["QViT"])

    # --- Core efficiency plots ---
    plot_acc_vs_params(records, str(out / "plots" / "acc_vs_params.png"))
    plot_acc_vs_epochs(histories, str(out / "plots" / "acc_vs_epochs.png"))
    plot_flops_vs_gates(records, str(out / "plots" / "flops_vs_gates.png"))
    plot_pareto_frontier(records, str(out / "plots" / "pareto_frontier.png"))
    plot_roc_pr([{"name": n, "probs": d["probs"], "labels": d["labels"]} for n, d in trained.items()],
                str(out / "plots" / "roc_pr.png"))

    # --- Embeddings (t-SNE) per model ---
    for name, d in trained.items():
        try:
            emb, lab = extract_embeddings(d["model"], test_loader, args.device)
            plot_embedding(project_2d(emb), lab, f"{name} feature space (t-SNE)",
                           str(out / "plots" / f"embeddings_{name}.png"))
        except Exception as e:  # noqa
            print(f"[warn] embedding plot failed for {name}: {e}")

    # --- Saliency comparison on a handful of test images ---
    for i in range(min(args.n_saliency, len(test_ds))):
        item = test_ds[i]
        x = item["image_classical"].unsqueeze(0).to(args.device)
        sal = {}
        for name, d in trained.items():
            try:
                sal[name] = model_saliency(d["model"], x)
            except Exception as e:  # noqa
                print(f"[warn] saliency failed for {name} img {i}: {e}")
        label = "defective" if item["label"].item() == 1 else "good"
        compare_models_figure(item["image_classical"], sal,
                              out_path=str(out / "plots" / f"saliency_compare_{i}.png"),
                              title=f"Sample {i} — ground truth: {label}")

    with open(out / "logs" / "benchmark.json", "w") as f:
        json.dump({"records": records, "histories": histories, "args": vars(args)}, f, indent=2)

    print("\n== Summary ==")
    print(f"{'model':>8s} | {'params':>9s} | {'val':>5s} | {'test':>5s} | {'f1':>5s} | {'auc':>5s} | {'gates':>8s}")
    for r in records:
        print(f"{r['model_name']:>8s} | {r['n_params']:>9,d} | {r['best_val_acc']:.3f} | "
              f"{r['test_acc']:.3f} | {r['test_f1']:.3f} | {r['test_auc']:.3f} | {r['quantum_gates']:>8d}")


if __name__ == "__main__":
    main()
