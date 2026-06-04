"""Training loops + Knowledge Distillation (classical ViT teacher -> QViT student).

KD formulation (Hinton et al., 2015):
    L = alpha * CE(student_logits, y)
      + (1 - alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))

Extras for practical, reproducible results:
  - label smoothing + gradient clipping + cosine schedule
  - best-checkpoint tracking (keeps the highest-val-acc weights)
  - optional two-phase 'quantum warm-up': train classical params first, then
    unfreeze the VQC params (mitigates wasting expensive quantum passes on a
    randomly-initialised backbone).
"""
from __future__ import annotations

import copy
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
    best_val_acc: float = 0.0
    best_state: Optional[dict] = None


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_quantum_param(name: str) -> bool:
    return ("qlayer" in name) or ("q_vqc" in name) or ("k_vqc" in name) or ("v_vqc" in name)


def set_quantum_requires_grad(model: nn.Module, flag: bool):
    for name, p in model.named_parameters():
        if is_quantum_param(name):
            p.requires_grad_(flag)


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


def _step_epoch(model, loader, optim, device, loss_fn, grad_clip):
    model.train()
    n, tot_loss, correct = 0, 0.0, 0
    for batch in loader:
        x = batch["image_classical"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        tot_loss += loss.item() * y.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.size(0)
    return tot_loss / n, correct / n


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    grad_clip: float = 1.0,
    quantum_warmup: int = 0,
    device: str = "cuda",
    model_name: str = "model",
    log_fn=print,
) -> RunHistory:
    model.to(device)
    hist = RunHistory(model_name=model_name, n_params=count_trainable_params(model))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if quantum_warmup > 0:
        set_quantum_requires_grad(model, False)
        log_fn(f"[{model_name}] quantum warm-up: VQC params frozen for {quantum_warmup} epoch(s)")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    for ep in range(1, epochs + 1):
        if quantum_warmup > 0 and ep == quantum_warmup + 1:
            set_quantum_requires_grad(model, True)
            optim = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs - quantum_warmup)
            log_fn(f"[{model_name}] quantum warm-up done: VQC params unfrozen")

        t0 = time.time()
        train_loss, train_acc = _step_epoch(model, train_loader, optim, device, loss_fn, grad_clip)
        sched.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        hist.history.append(EpochMetrics(ep, train_loss, train_acc, val_loss, val_acc, time.time() - t0))
        if val_acc > hist.best_val_acc:
            hist.best_val_acc = val_acc
            hist.best_state = copy.deepcopy(model.state_dict())
        log_fn(f"[{model_name}] ep {ep:03d} | train {train_acc:.3f} ({train_loss:.3f}) "
               f"| val {val_acc:.3f} ({val_loss:.3f}) | {hist.history[-1].seconds:.1f}s")

    if hist.best_state is not None:
        model.load_state_dict(hist.best_state)  # restore best weights
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
    label_smoothing: float = 0.05,
    grad_clip: float = 1.0,
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
    ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
            ce = ce_fn(s_logits, y)
            kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            loss = alpha * ce + (1 - alpha) * kd
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optim.step()
            tot_loss += loss.item() * y.size(0)
            correct += (s_logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        sched.step()
        train_loss, train_acc = tot_loss / n, correct / n
        val_loss, val_acc = evaluate(student, val_loader, device)
        hist.history.append(EpochMetrics(ep, train_loss, train_acc, val_loss, val_acc, time.time() - t0))
        if val_acc > hist.best_val_acc:
            hist.best_val_acc = val_acc
            hist.best_state = copy.deepcopy(student.state_dict())
        log_fn(f"[{model_name} KD] ep {ep:03d} | train {train_acc:.3f} ({train_loss:.3f}) "
               f"| val {val_acc:.3f} ({val_loss:.3f}) | {hist.history[-1].seconds:.1f}s")

    if hist.best_state is not None:
        student.load_state_dict(hist.best_state)
    return hist


def history_to_dict(h: RunHistory) -> Dict:
    return {
        "model_name": h.model_name,
        "n_params": h.n_params,
        "best_val_acc": h.best_val_acc,
        "history": [m.__dict__ for m in h.history],
    }
