"""Augmentation-sensitivity probe.

The class_aware experiment in findings.md Section 3c failed because my
hand-picked partition of "rotation-safe" vs "rotation-sensitive" classes
was wrong. This module replaces guesswork with measurement.

For each training-time augmentation transform t and each class c, we apply
t to every test image of class c, run the *trained* baseline model on both
the original and the augmented copy, and measure how much the model's
predicted probability for the true class drops:

    sensitivity(c, t) = mean over images i of class c of
                        max(0, p_true(x_i) - p_true(t(x_i)))

A high sensitivity score means the transform destroys class signal — the
model can't recognise the augmented version as the same class. Apply the
transform to that class at training time and you're injecting noise into
the loss, not invariance.

We then auto-derive an augmentation policy: rotate only the classes whose
rotation-sensitivity is below a chosen percentile of the per-class scores.

Run with:
    python -m src.sensitivity --run runs/baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .data import CLASS_NAMES, NEUCLS, NUM_CLASSES
from .model import IMAGENET_MEAN, IMAGENET_STD, build_model, build_transform

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _to_tensor_norm(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((224, 224))
    t = TF.to_tensor(img)
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


# Each transform takes a PIL image and returns a PIL image. We deliberately
# operate at PIL level (not on tensors) so the augmentations are identical
# to the train-time ones in src/model.py:build_transform.
TRANSFORMS = {
    "h_flip": lambda img: TF.hflip(img),
    "rot_5": lambda img: TF.rotate(img, 5),
    "rot_15": lambda img: TF.rotate(img, 15),
    "rot_30": lambda img: TF.rotate(img, 30),
}


def _load_model(run_dir: Path) -> torch.nn.Module:
    device = torch.device("cpu")
    model = build_model().to(device)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _probs(model: torch.nn.Module, img: Image.Image) -> np.ndarray:
    x = _to_tensor_norm(img).unsqueeze(0)
    return F.softmax(model(x), dim=1).cpu().numpy()[0]


def measure_sensitivity(model: torch.nn.Module, dataset: NEUCLS, n_per_class: int | None = None) -> pd.DataFrame:
    """For every (class, transform) cell, compute mean p_true drop."""
    rows = []
    by_class: dict[int, list[int]] = {c: [] for c in range(NUM_CLASSES)}
    for i, lab in enumerate(dataset.labels):
        by_class[int(lab)].append(i)

    for c in range(NUM_CLASSES):
        idx_list = by_class[c]
        if n_per_class is not None:
            idx_list = idx_list[:n_per_class]
        for t_name, t_fn in TRANSFORMS.items():
            drops = []
            for idx in idx_list:
                img = dataset.get_pil(idx)
                p_orig = _probs(model, img)[c]
                p_aug = _probs(model, t_fn(img))[c]
                drops.append(max(0.0, float(p_orig - p_aug)))
            rows.append(
                {
                    "class": CLASS_NAMES[c],
                    "transform": t_name,
                    "n": len(idx_list),
                    "mean_p_true_drop": float(np.mean(drops)),
                    "p95_p_true_drop": float(np.percentile(drops, 95)) if drops else 0.0,
                }
            )
    return pd.DataFrame(rows)


def derive_policy(df: pd.DataFrame, transform: str = "rot_15", percentile: float = 50.0) -> dict:
    """Pick the classes whose sensitivity to `transform` is below the
    `percentile` cutoff — those are the rotation-safe classes that should
    receive the transform at training time."""
    sub = df[df["transform"] == transform].set_index("class")
    cutoff = float(np.percentile(sub["mean_p_true_drop"], percentile))
    safe = sub.index[sub["mean_p_true_drop"] <= cutoff].tolist()
    sensitive = sub.index[sub["mean_p_true_drop"] > cutoff].tolist()
    return {
        "transform": transform,
        "percentile_cutoff": percentile,
        "cutoff_value": cutoff,
        "safe_classes": safe,
        "sensitive_classes": sensitive,
        "per_class_drop": sub["mean_p_true_drop"].round(4).to_dict(),
    }


def heatmap(df: pd.DataFrame, out_png: Path) -> None:
    pivot = df.pivot(index="class", columns="transform", values="mean_p_true_drop")
    # Reorder transforms: flip first, then rotations by magnitude.
    cols = [c for c in ["h_flip", "rot_5", "rot_15", "rot_30"] if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=max(0.1, pivot.values.max()))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=15)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > pivot.values.max() / 2 else "black", fontsize=9)
    ax.set_title("Augmentation sensitivity by class\n(mean drop in p(true class) after transform)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to a trained run (uses its model.pt)")
    parser.add_argument("--transform", default="rot_15", choices=list(TRANSFORMS.keys()))
    parser.add_argument("--percentile", type=float, default=50.0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = ROOT / "reports" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {run_dir}/model.pt ...")
    model = _load_model(run_dir)
    print(f"Probing on the test set across {len(TRANSFORMS)} transforms ...")
    test = NEUCLS(DATA_DIR / "test.parquet", transform=None)
    df = measure_sensitivity(model, test)
    df.to_csv(out_dir / "per_class_sensitivity.csv", index=False)
    heatmap(df, out_dir / "per_class_sensitivity.png")

    policy = derive_policy(df, transform=args.transform, percentile=args.percentile)
    (out_dir / "policy.json").write_text(json.dumps(policy, indent=2))
    print()
    print(f"=== Sensitivity to {args.transform} (median cut at p{args.percentile:.0f}) ===")
    for cls, drop in sorted(policy["per_class_drop"].items(), key=lambda kv: kv[1]):
        marker = "SAFE  " if cls in policy["safe_classes"] else "SENSI "
        print(f"  {marker} {cls:<18}  drop={drop:.4f}")
    print()
    print("Recommended policy:")
    print(f"  apply {args.transform} at training time only to: {policy['safe_classes']}")
    print(f"  skip it for: {policy['sensitive_classes']}")


if __name__ == "__main__":
    main()
