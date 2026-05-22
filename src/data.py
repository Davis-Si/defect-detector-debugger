"""NEU-CLS data loading.

The dataset on Hugging Face (`newguyme/neu_cls`) ships as parquet with two
columns: `image` (encoded bytes wrapped in {bytes, path}) and `label` (int).
There are 1440 train and 360 test images across 6 classes of hot-rolled steel
surface defects. Images are 200x200 grayscale, stored as PNG/JPEG bytes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class Sample:
    image: Image.Image
    label: int
    index: int  # row index inside its split — used to track examples through analyses


def _decode(cell) -> Image.Image:
    """parquet image cell -> PIL.Image. Handles dict-with-bytes and raw bytes."""
    if isinstance(cell, dict):
        raw = cell.get("bytes")
    else:
        raw = cell
    return Image.open(io.BytesIO(raw)).convert("RGB")


class NEUCLS(Dataset):
    def __init__(self, parquet_path: str | Path, transform=None, *, in_memory: bool = True):
        self.parquet_path = Path(parquet_path)
        self.transform = transform
        table = pq.read_table(self.parquet_path)
        df = table.to_pandas()
        self.labels = df["label"].to_numpy().astype(np.int64)
        self._raw = df["image"].tolist()
        self._cache: list[Image.Image] | None = None
        if in_memory:
            self._cache = [_decode(c) for c in self._raw]

    def __len__(self) -> int:
        return len(self.labels)

    def get_pil(self, idx: int) -> Image.Image:
        if self._cache is not None:
            return self._cache[idx]
        return _decode(self._raw[idx])

    def __getitem__(self, idx: int):
        img = self.get_pil(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx]), idx


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}


def stratified_train_val_split(labels: np.ndarray, val_frac: float = 0.15, seed: int = 0):
    """Return (train_idx, val_idx) with class-balanced sampling."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_frac)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))
