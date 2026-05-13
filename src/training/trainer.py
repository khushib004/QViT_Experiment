"""Training loops + Knowledge Distillation (classical ViT teacher -> QViT student).

KD formulation (Hinton et al., 2015):
    L = alpha * CE(student_logits, y)
      + (1 - alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    seconds: float


@dataclass
class RunHistory:
    model_name: str
    n_params: int
    history: List[EpochMetrics] = field(default_factory=list)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    n, total_loss, correct = 0, 0.0, 0
    for batch in loader:
        x = batch["image_classical"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    model_name: str = "model",
    log_fn=print,
) -> RunHistory:
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    hist = RunHistory(model_name=model_name, n_params=count_trainable_params(model))

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        n, tot_loss, correct = 0, 0.0, 0
        for batch in train_loader:
            x = batch["image_classical"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        sched.step()
        train_loss, train_acc = tot_loss / n, correct / n
        val_loss, val_acc = evaluate(model, val_loader, device)
        m = EpochMetrics(ep, train_loss, train_acc, val_loss, val_acc, time.time() - t0)
        hist.history.append(m)
        log_fn(
            f"[{model_name}] ep {ep:03d} | train {train_acc:.3f} ({train_loss:.3f}) "
            f"| val {val_acc:.3f} ({val_loss:.3f}) | {m.seconds:.1f}s"
        )
    return hist


def train_distilled(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    device: str = "cuda",
    model_name: str = "qvit_distilled",
    log_fn=print,
) -> RunHistory:
    """Train `student` with KD loss against a frozen `teacher`."""
    student.to(device)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    hist = RunHistory(model_name=model_name, n_params=count_trainable_params(student))
    T = temperature

    for ep in range(1, epochs + 1):
        student.train()
        t0 = time.time()
        n, tot_loss, correct = 0, 0.0, 0
        for batch in train_loader:
            x = batch["image_classical"].to(device)
            y = batch["label"].to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)

            ce = F.cross_entropy(s_logits, y)
            kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            loss = alpha * ce + (1 - alpha) * kd

            optim.zero_grad()
            loss.backward()
            optim.step()
            tot_loss += loss.item() * y.size(0)
            correct += (s_logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        sched.step()
        train_loss, train_acc = tot_loss / n, correct / n
        val_loss, val_acc = evaluate(student, val_loader, device)
        m = EpochMetrics(ep, train_loss, train_acc, val_loss, val_acc, time.time() - t0)
        hist.history.append(m)
        log_fn(
            f"[{model_name} KD] ep {ep:03d} | train {train_acc:.3f} ({train_loss:.3f}) "
            f"| val {val_acc:.3f} ({val_loss:.3f}) | {m.seconds:.1f}s"
        )
    return hist


def history_to_dict(h: RunHistory) -> Dict:
    return {
        "model_name": h.model_name,
        "n_params": h.n_params,
        "history": [m.__dict__ for m in h.history],
    }
