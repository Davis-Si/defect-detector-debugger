"""Coverage-vs-accuracy analysis: how to ship this model.

A production inspection system rarely deploys a CV model with no escape hatch.
The standard pattern is:

  if model.confidence(x) >= threshold T: auto-decide
  else:                                   route to human review

This script sweeps T over the test set, computes:

  coverage(T)  = fraction of frames the model auto-decides
  accuracy(T)  = accuracy of those auto-decisions
  defer_rate   = 1 - coverage

and produces three artefacts:
  reports/deployment/coverage_accuracy.csv   the full sweep
  reports/deployment/coverage_accuracy.png   the curve
  reports/deployment/recommendation.json     the recommended threshold

The recommended threshold is the smallest T at which the auto-decided slice
is 100% accurate on the test set — i.e. the smallest threshold that lets you
ship without a single auto-decision being wrong on this evaluation. Defer
rate at that threshold tells operations how much manual capacity they need.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sweep(preds: pd.DataFrame, n_thresholds: int = 41) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []
    correct = (preds["label"] == preds["pred"]).to_numpy()
    conf = preds["confidence"].to_numpy()
    for t in thresholds:
        mask = conf >= t
        n_auto = int(mask.sum())
        coverage = n_auto / len(preds)
        if n_auto == 0:
            acc, errors_in_auto = 1.0, 0
        else:
            acc = float(correct[mask].mean())
            errors_in_auto = int((~correct[mask]).sum())
        rows.append(
            {
                "threshold": float(t),
                "coverage": coverage,
                "n_auto_decided": n_auto,
                "auto_accuracy": acc,
                "auto_errors": errors_in_auto,
                "defer_rate": 1.0 - coverage,
            }
        )
    return pd.DataFrame(rows)


def plot_curve(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(df["threshold"], df["auto_accuracy"], "o-", color="#264653", label="auto-decision accuracy")
    ax1.set_xlabel("confidence threshold T")
    ax1.set_ylabel("auto-decision accuracy", color="#264653")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#264653")
    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["coverage"], "s--", color="#e76f51", label="coverage")
    ax2.set_ylabel("coverage (fraction auto-decided)", color="#e76f51")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#e76f51")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def recommend(df: pd.DataFrame) -> dict:
    """Lowest threshold at which auto-decisions are 100% correct."""
    perfect = df[df["auto_errors"] == 0].sort_values("threshold")
    if len(perfect) == 0:
        return {"feasible": False}
    # Prefer the lowest threshold (= highest coverage) that is still perfect.
    chosen = perfect.iloc[0]
    return {
        "feasible": True,
        "threshold": float(chosen["threshold"]),
        "coverage": float(chosen["coverage"]),
        "defer_rate": float(chosen["defer_rate"]),
        "auto_accuracy": float(chosen["auto_accuracy"]),
        "n_auto_decided": int(chosen["n_auto_decided"]),
        "auto_errors": int(chosen["auto_errors"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to a run directory")
    args = parser.parse_args()
    run_dir = Path(args.run)
    preds = pd.read_csv(run_dir / "test_predictions.csv")

    out_dir = ROOT / "reports" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = sweep(preds)
    df.to_csv(out_dir / "coverage_accuracy.csv", index=False)
    plot_curve(df, out_dir / "coverage_accuracy.png", f"Deployment policy — {run_dir.name}")
    rec = recommend(df)
    rec["run"] = run_dir.name
    rec["test_set_size"] = len(preds)
    (out_dir / "recommendation.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
