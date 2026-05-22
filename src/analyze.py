"""Failure-analysis toolkit for a NEU-CLS classifier run.

Generates the artefacts that hiring teams in applied CV actually read:
  1. Per-class precision / recall / F1 (CSV + bar plot).
  2. Confusion matrix (PNG).
  3. Confidence-binned accuracy + a calibration curve.
  4. Most-confused class-pair galleries (one PNG per top pair).
  5. Hard examples board: high-confidence wrong + low-confidence right.
  6. t-SNE of penultimate features, coloured by class and by correctness.
  7. Run comparison table across augmentation ablations.

Usage:
    python -m src.analyze --runs runs/baseline runs/flip runs/flip_rotate
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import CLASS_NAMES, NEUCLS, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_run(run_dir: Path):
    preds = pd.read_csv(run_dir / "test_predictions.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    feats = np.load(run_dir / "test_features.npy")
    return preds, metrics, feats


def per_class_table(preds: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        preds["label"], preds["pred"], labels=list(range(NUM_CLASSES)), zero_division=0
    )
    df = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "support": support,
            "precision": p.round(4),
            "recall": r.round(4),
            "f1": f1.round(4),
        }
    )
    df.to_csv(out_csv, index=False)
    return df


def plot_per_class(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df["precision"], w, label="precision")
    ax.bar(x, df["recall"], w, label="recall")
    ax.bar(x + w, df["f1"], w, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_confusion(preds: pd.DataFrame, out_png: Path, title: str) -> np.ndarray:
    cm = confusion_matrix(preds["label"], preds["pred"], labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return cm


def confidence_slice(preds: pd.DataFrame, out_png: Path, title: str) -> pd.DataFrame:
    bins = np.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.0001])
    preds = preds.copy()
    preds["bin"] = pd.cut(preds["confidence"], bins=bins, include_lowest=True, right=False)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    grp = preds.groupby("bin", observed=True).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_conf=("confidence", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(grp["mean_conf"], grp["accuracy"], "o-", label="model")
    for _, row in grp.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["mean_conf"], row["accuracy"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean predicted confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return grp


def _image_grid(images: list[Image.Image], titles: list[str], out_png: Path, suptitle: str) -> None:
    if not images:
        return
    n = len(images)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], cmap="gray")
            ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def confused_pair_galleries(
    preds: pd.DataFrame, cm: np.ndarray, test_pil: NEUCLS, out_dir: Path, top_k: int = 3
) -> list[tuple[int, int, int]]:
    """For the top-K most-confused (true, pred) pairs, save a gallery of examples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    off_diag = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                off_diag.append((cm[i, j], i, j))
    off_diag.sort(reverse=True)
    top = off_diag[:top_k]
    for count, true_c, pred_c in top:
        rows = preds[(preds["label"] == true_c) & (preds["pred"] == pred_c)]
        rows = rows.sort_values("confidence", ascending=False).head(8)
        imgs = [test_pil.get_pil(int(r.index)) for r in rows.itertuples()]
        titles = [f"conf={r.confidence:.2f}" for r in rows.itertuples()]
        suptitle = (
            f"True: {CLASS_NAMES[true_c]}  |  Predicted: {CLASS_NAMES[pred_c]}  "
            f"(n={count}, sorted by model confidence)"
        )
        out = out_dir / f"confused__{CLASS_NAMES[true_c]}__as__{CLASS_NAMES[pred_c]}.png"
        _image_grid(imgs, titles, out, suptitle)
    return [(true_c, pred_c, count) for count, true_c, pred_c in top]


def hard_examples(preds: pd.DataFrame, test_pil: NEUCLS, out_dir: Path) -> dict:
    """Save two boards: high-confidence wrong, and low-confidence right."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrong = preds[preds["label"] != preds["pred"]].sort_values("confidence", ascending=False).head(12)
    right = preds[preds["label"] == preds["pred"]].sort_values("confidence", ascending=True).head(12)

    imgs = [test_pil.get_pil(int(r.index)) for r in wrong.itertuples()]
    titles = [
        f"true={CLASS_NAMES[r.label]}\npred={CLASS_NAMES[r.pred]} ({r.confidence:.2f})"
        for r in wrong.itertuples()
    ]
    _image_grid(
        imgs,
        titles,
        out_dir / "high_conf_wrong.png",
        "High-confidence WRONG predictions (most likely label/data issues)",
    )

    imgs = [test_pil.get_pil(int(r.index)) for r in right.itertuples()]
    titles = [
        f"true={CLASS_NAMES[r.label]}\nconf={r.confidence:.2f}, 2nd={r.second_confidence:.2f}"
        for r in right.itertuples()
    ]
    _image_grid(
        imgs,
        titles,
        out_dir / "low_conf_right.png",
        "Low-confidence CORRECT predictions (genuinely ambiguous samples)",
    )

    return {
        "high_conf_wrong_count": int((preds["label"] != preds["pred"]).sum()),
        "lowest_correct_conf": float(right["confidence"].min()) if len(right) else None,
        "highest_wrong_conf": float(wrong["confidence"].max()) if len(wrong) else None,
    }


def tsne_plot(features: np.ndarray, preds: pd.DataFrame, out_png: Path, title: str) -> None:
    n = len(features)
    perplexity = max(5, min(30, n // 6))
    emb = TSNE(
        n_components=2, init="pca", perplexity=perplexity, random_state=0, learning_rate="auto"
    ).fit_transform(features)
    correct = (preds["label"] == preds["pred"]).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    palette = sns.color_palette("tab10", NUM_CLASSES)
    for c in range(NUM_CLASSES):
        m = preds["label"] == c
        axes[0].scatter(emb[m, 0], emb[m, 1], s=14, color=palette[c], label=CLASS_NAMES[c], alpha=0.8)
    axes[0].set_title("t-SNE of test features — by true class")
    axes[0].legend(loc="best", fontsize=8, markerscale=1.5)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].scatter(emb[correct, 0], emb[correct, 1], s=14, color="#2a9d8f", label="correct", alpha=0.7)
    axes[1].scatter(emb[~correct, 0], emb[~correct, 1], s=22, color="#e63946", label="wrong", alpha=0.95, marker="x")
    axes[1].set_title("t-SNE of test features — by correctness")
    axes[1].legend(loc="best")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_comparison(runs: list[Path], out_csv: Path, out_png: Path) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = json.loads((r / "metrics.json").read_text())
        cfg = json.loads((r / "config.json").read_text())
        rows.append(
            {
                "run": r.name,
                "augment": cfg.get("augment"),
                "epochs": cfg.get("epochs"),
                "best_val_acc": round(m["best_val_acc"], 4),
                "test_acc": round(m["test_acc"], 4),
                "test_loss": round(m["test_loss"], 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(df["run"], df["test_acc"], color="#264653")
    for i, v in enumerate(df["test_acc"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Augmentation ablation: test accuracy by run")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return df


def analyse_run(run_dir: Path, test_pil: NEUCLS) -> dict:
    preds, metrics, feats = load_run(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    df = per_class_table(preds, out / "per_class.csv")
    plot_per_class(df, out / "per_class.png", f"Per-class metrics — {run_dir.name}")
    cm = plot_confusion(preds, out / "confusion_matrix.png", f"Confusion matrix — {run_dir.name}")
    conf_table = confidence_slice(preds, out / "calibration.png", f"Calibration — {run_dir.name}")
    conf_table.to_csv(out / "confidence_bins.csv", index=False)
    confused = confused_pair_galleries(preds, cm, test_pil, out / "confused_pairs", top_k=3)
    hard = hard_examples(preds, test_pil, out / "hard_examples")
    tsne_plot(feats, preds, out / "tsne.png", f"t-SNE — {run_dir.name}")

    summary = {
        "run": run_dir.name,
        "test_acc": metrics["test_acc"],
        "per_class": df.to_dict(orient="records"),
        "top_confused_pairs": [
            {"true": CLASS_NAMES[t], "pred": CLASS_NAMES[p], "count": int(c)} for t, p, c in confused
        ],
        "hard_examples_summary": hard,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="paths to run directories")
    args = parser.parse_args()

    test_pil = NEUCLS(DATA_DIR / "test.parquet", transform=None)

    summaries = []
    for r in args.runs:
        run_dir = Path(r)
        print(f"\n=== Analysing {run_dir.name} ===")
        summaries.append(analyse_run(run_dir, test_pil))

    # Cross-run comparison table.
    runs_root = Path(args.runs[0]).parent
    cmp_dir = runs_root.parent / "reports"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    df = run_comparison([Path(r) for r in args.runs], cmp_dir / "ablation.csv", cmp_dir / "ablation.png")
    print("\n=== Ablation summary ===")
    print(df.to_string(index=False))

    (cmp_dir / "summaries.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
