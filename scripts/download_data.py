"""Download the TyreNet dataset from Kaggle and normalise the folder layout.

Source: https://www.kaggle.com/datasets/warcoder/tyre-quality-classification
After download, this script ensures the dataset lives at
    <dest_root>/good/*.jpg
    <dest_root>/defective/*.jpg
which is what `src.data.TyreNetDataset` expects.

Kaggle auth options (any one):
  - In Colab: Settings -> Secrets -> add KAGGLE_USERNAME and KAGGLE_KEY.
  - On desktop: place your kaggle.json at ~/.kaggle/kaggle.json (chmod 600).
  - Export env vars: KAGGLE_USERNAME=..., KAGGLE_KEY=...

Usage:
    python scripts/download_data.py                 # downloads to ./data/tyrenet
    python scripts/download_data.py --dest /tmp/td  # custom destination
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DATASET_SLUG = "warcoder/tyre-quality-classification"


def _ensure_kagglehub():
    try:
        import kagglehub  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])


def _propagate_colab_secrets():
    """If running in Colab and KAGGLE_USERNAME/KAGGLE_KEY are stored as secrets,
    surface them as env vars so kagglehub picks them up automatically."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if k in os.environ:
            continue
        try:
            v = userdata.get(k)
            if v:
                os.environ[k] = v
        except Exception:
            pass


def _find_class_dirs(root: Path) -> dict:
    """Recursively find the 'good' and 'defective' folders in the download."""
    matches = {"good": None, "defective": None}
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        n = p.name.lower()
        if n in matches and matches[n] is None:
            matches[n] = p
    return matches


def _materialise(src: Path, dest: Path, mode: str = "symlink"):
    """Place the source folder at `dest`. Prefer symlinks (fast, no disk use),
    fall back to copy if the filesystem does not support symlinks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    if mode == "symlink":
        try:
            dest.symlink_to(src.resolve(), target_is_directory=True)
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copytree(src, dest)


def download_tyrenet(dest_root: str = "data/tyrenet") -> Path:
    """Download (if needed) and link the dataset into `dest_root`."""
    dest = Path(dest_root)
    if (dest / "good").exists() and (dest / "defective").exists():
        print(f"[ok] dataset already present at {dest.resolve()}")
        return dest

    _ensure_kagglehub()
    _propagate_colab_secrets()
    import kagglehub  # noqa: E402

    print(f"[kagglehub] downloading {DATASET_SLUG} ...")
    src = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"[kagglehub] cached at: {src}")

    cls_dirs = _find_class_dirs(src)
    missing = [k for k, v in cls_dirs.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Could not find {missing} folders inside {src}. "
            f"Contents: {[p.name for p in src.iterdir()]}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    for cls, p in cls_dirs.items():
        target = dest / cls
        _materialise(p, target)
        n_files = sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"[ok] {cls:>9s}: {n_files:>5d} images -> {target}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/tyrenet")
    args = ap.parse_args()
    download_tyrenet(args.dest)


if __name__ == "__main__":
    main()
