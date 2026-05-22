"""INT8 dynamic quantization + accuracy/latency Pareto.

Quantizes the trained baseline using PyTorch eager-mode dynamic quantization
on the Linear head. (Conv layers in a frozen ResNet-18 backbone gain little
from dynamic quant; serious INT8 conv gains require static quant with
calibration data, which is heavier than the rest of this project.)

Even with just the head quantized, the model size shrinks slightly and the
small linear path becomes faster. The real product of this script is the
*Pareto frontier plot* combining latency and accuracy across:
  - fp32 baseline
  - int8 dynamic quantization

So a reviewer can see exactly what trade you're making for the speedup.

Usage:
    python -m src.quantize --run runs/baseline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader

from .data import NEUCLS
from .model import build_model, build_transform

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def model_size_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.state_dict().values():
        if hasattr(p, "numel"):
            # Try element_size first, fall back for quantized tensors.
            try:
                total += p.numel() * p.element_size()
            except (AttributeError, RuntimeError):
                total += p.numel()
    return total


def _evaluate(model: nn.Module, loader) -> tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return loss_sum / total, correct / total


def _bench(model: nn.Module, batch_size: int, n_iters: int = 30, warmup: int = 5) -> dict:
    x = torch.randn(batch_size, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return {
        "batch_size": batch_size,
        "p50_batch_ms": float(np.percentile(arr, 50) * 1000),
        "p95_batch_ms": float(np.percentile(arr, 95) * 1000),
        "p50_per_image_ms": float(np.percentile(arr / batch_size, 50) * 1000),
        "throughput_fps": float(batch_size / np.median(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)

    out_dir = ROOT / "reports" / "quantization"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load fp32 baseline.
    fp32 = build_model()
    state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    fp32.load_state_dict(state)

    # Quantize dynamically. Conv2d isn't supported for dynamic quant in eager
    # mode, but Linear is — so we target {nn.Linear}.
    int8 = quantize_dynamic(fp32, qconfig_spec={nn.Linear}, dtype=torch.qint8)

    # Test loaders.
    eval_tf = build_transform(train=False)
    test = NEUCLS(DATA_DIR / "test.parquet", transform=eval_tf)
    loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)

    print("Evaluating fp32 ...")
    fp32_loss, fp32_acc = _evaluate(fp32, loader)
    print("Evaluating int8 (dynamic) ...")
    int8_loss, int8_acc = _evaluate(int8, loader)

    rows = []
    for variant, model in (("fp32", fp32), ("int8_dynamic", int8)):
        print(f"\n=== Benchmarking {variant} ===")
        size = model_size_bytes(model)
        for bs in (1, 4, 16, 32):
            r = _bench(model, batch_size=bs)
            rows.append(
                {
                    "variant": variant,
                    "size_MB": round(size / 1024 / 1024, 2),
                    **r,
                }
            )
            print(
                f"  batch={bs:>2}  p50={r['p50_batch_ms']:>7.2f}ms  "
                f"per-image={r['p50_per_image_ms']:>5.2f}ms  fps={r['throughput_fps']:>5.1f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark.csv", index=False)

    summary = {
        "fp32": {"test_loss": fp32_loss, "test_acc": fp32_acc, "size_MB": round(model_size_bytes(fp32) / 1024 / 1024, 2)},
        "int8_dynamic": {"test_loss": int8_loss, "test_acc": int8_acc, "size_MB": round(model_size_bytes(int8) / 1024 / 1024, 2)},
        "accuracy_delta": round(int8_acc - fp32_acc, 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Pareto plot: per-image latency at batch=1 vs test accuracy.
    bs1 = df[df["batch_size"] == 1].set_index("variant")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    accs = {"fp32": fp32_acc, "int8_dynamic": int8_acc}
    for v, color in (("fp32", "#264653"), ("int8_dynamic", "#e76f51")):
        ax.scatter(
            bs1.loc[v, "p50_per_image_ms"],
            accs[v],
            s=170,
            color=color,
            label=f"{v}: {accs[v]:.4f} acc, {bs1.loc[v, 'p50_per_image_ms']:.1f} ms/img",
            edgecolors="black",
        )
    ax.set_xlabel("p50 latency per image (ms, batch=1)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs latency Pareto — fp32 vs INT8 dynamic")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=140)
    plt.close(fig)

    print("\n=== Pareto summary ===")
    print(f"  fp32:        acc={fp32_acc:.4f}  size={summary['fp32']['size_MB']:.2f} MB")
    print(f"  int8_dyn:    acc={int8_acc:.4f}  size={summary['int8_dynamic']['size_MB']:.2f} MB")
    print(f"  accuracy delta: {summary['accuracy_delta']:+.4f}")


if __name__ == "__main__":
    main()
