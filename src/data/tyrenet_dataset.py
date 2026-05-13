"""
TyreNet (Mendeley) dataset wrapper for industrial tyre defect classification.

Two-resolution output (per sample):
    - 'image_classical': (3, 224, 224) float tensor for CNN / ViT / QCNN baseline.
    - 'image_quantum':   (P, n_qubits) float tensor — per-patch reduced features
      for quantum models, produced by either PCA or a strided conv reducer.
    - 'label':           {0: good, 1: defective}

Directory layout expected (case-insensitive):
    root/
        good/        *.jpg|png|...
        defective/   *.jpg|png|...
(or a single CSV with columns image_path,label)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from sklearn.decomposition import PCA  # optional path
except ImportError:  # pragma: no cover
    PCA = None


CLASS_MAP = {"good": 0, "defective": 1, "defect": 1, "bad": 1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _discover_samples(root: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    root_p = Path(root)
    for cls_dir in sorted(p for p in root_p.iterdir() if p.is_dir()):
        key = cls_dir.name.lower()
        if key not in CLASS_MAP:
            continue
        label = CLASS_MAP[key]
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((str(f), label))
    if not samples:
        raise RuntimeError(f"No images found under {root}. Expected good/ and defective/ subfolders.")
    return samples


class StridedConvReducer(torch.nn.Module):
    """A frozen strided-conv 'quantum-ready' patch reducer.

    Slices the 224x224 image into a grid of patches and emits an n_qubits-dim
    descriptor per patch using a single Conv2d. Channels are tanh-squashed to
    [-1, 1] so they are valid AngleEmbedding inputs.
    """

    def __init__(self, n_qubits: int = 4, patch_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=n_qubits,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        # Use a fixed orthogonal init so reducer is deterministic across runs
        torch.nn.init.orthogonal_(self.conv.weight)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (3, H, W) -> (n_qubits, H/ps, W/ps) -> (P, n_qubits)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        feats = self.conv(img)
        B, C, H, W = feats.shape
        return torch.tanh(feats.view(B, C, H * W).permute(0, 2, 1)).squeeze(0)


class PCAReducer:
    """Image -> per-patch PCA descriptor.

    Patches are flattened to (P, patch_size*patch_size*3) and projected through
    a pre-fitted PCA. Output is tanh-normalised to [-1, 1].
    Fit once on the training split with `PCAReducer.fit(...)`.
    """

    def __init__(self, n_components: int = 4, patch_size: int = 32):
        if PCA is None:
            raise ImportError("scikit-learn is required for PCAReducer.")
        self.pca = PCA(n_components=n_components)
        self.patch_size = patch_size
        self.n_components = n_components
        self._fitted = False

    def _patchify(self, img_np: np.ndarray) -> np.ndarray:
        # img_np: (H, W, 3) uint8 or float
        H, W, C = img_np.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0
        patches = (
            img_np.reshape(H // ps, ps, W // ps, ps, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(-1, ps * ps * C)
        )
        return patches.astype(np.float32) / 255.0

    def fit(self, image_paths: List[str], max_images: int = 200) -> "PCAReducer":
        pool: List[np.ndarray] = []
        resize = T.Compose([T.Resize((224, 224))])
        for p in image_paths[:max_images]:
            img = resize(Image.open(p).convert("RGB"))
            pool.append(self._patchify(np.array(img)))
        X = np.concatenate(pool, axis=0)
        self.pca.fit(X)
        self._fitted = True
        return self

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("PCAReducer must be .fit() before use.")
        # img is a normalised (3,224,224) tensor in roughly [-2.5, 2.5];
        # de-normalise into [0,1] image space first.
        np_img = img.detach().cpu().permute(1, 2, 0).numpy()
        np_img = (np_img - np_img.min()) / (np_img.ptp() + 1e-8)
        patches = self._patchify((np_img * 255).astype(np.uint8))
        proj = self.pca.transform(patches).astype(np.float32)
        return torch.from_numpy(np.tanh(proj))


def default_classical_transform(train: bool) -> Callable:
    if train:
        return T.Compose(
            [
                T.Resize((232, 232)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.1, 0.1, 0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class TyreNetDataset(Dataset):
    """Dual-resolution TyreNet dataset.

    Args:
        root: dataset root with good/ and defective/ subfolders.
        split: 'train' | 'val' | 'test' (file-level random split via `split_seed`).
        train_frac, val_frac: dataset split fractions.
        n_qubits: feature dimension for quantum reducer.
        patch_size: tile size for the quantum patch reducer (224/patch_size grid).
        reducer: 'conv' (default, fast) or 'pca' (requires .fit on train split).
        return_quantum: if False, skip the quantum tensor (saves time for CNN/ViT).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        split_seed: int = 42,
        n_qubits: int = 4,
        patch_size: int = 32,
        reducer: str = "conv",
        return_quantum: bool = True,
        classical_transform: Optional[Callable] = None,
    ):
        super().__init__()
        samples = _discover_samples(root)
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(len(samples))
        n_train = int(len(samples) * train_frac)
        n_val = int(len(samples) * val_frac)
        splits = {
            "train": idx[:n_train],
            "val": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }
        if split not in splits:
            raise ValueError(f"split must be one of {list(splits)}")
        self.samples = [samples[i] for i in splits[split]]
        self.split = split
        self.return_quantum = return_quantum
        self.n_qubits = n_qubits
        self.patch_size = patch_size

        self.classical_tf = classical_transform or default_classical_transform(split == "train")

        if reducer == "conv":
            self.reducer: Callable = StridedConvReducer(n_qubits=n_qubits, patch_size=patch_size)
        elif reducer == "pca":
            self.reducer = PCAReducer(n_components=n_qubits, patch_size=patch_size).fit(
                [s[0] for s in self.samples]
            )
        else:
            raise ValueError("reducer must be 'conv' or 'pca'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        x_cls = self.classical_tf(img)
        item = {"image_classical": x_cls, "label": torch.tensor(label, dtype=torch.long)}
        if self.return_quantum:
            item["image_quantum"] = self.reducer(x_cls)  # (P, n_qubits)
        return item

    @property
    def num_patches(self) -> int:
        return (224 // self.patch_size) ** 2
