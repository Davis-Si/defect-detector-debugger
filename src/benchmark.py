"""CPU inference-latency benchmark.

Why this is here: a defect classifier that runs at 0.5 FPS is useless on a
production line that emits 30 FPS. Cerrion deploys to factory floors, so the
deploy-readiness signal we want from the model is wallclock per-frame
inference time on CPU — not just accuracy. This script measures it across
batch sizes, for both single-stream and batched-stream regimes.

Usage:
    python -m src.benchmark --run runs/baseline
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .model import build_model

ROOT = Path(__file__).resolve().parents[1]


def _bench(model: torch.nn.Module, batch_size: int, n_iters: int, warmup: int) -> dict:
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
    per_image = arr / batch_size
    return {
        "batch_size": batch_size,
        "n_iters": n_iters,
        "p50_batch_ms": float(np.percentile(arr, 50) * 1000),
        "p95_batch_ms": float(np.percentile(arr, 95) * 1000),
        "p50_per_image_ms": float(np.percentile(per_image, 50) * 1000),
        "p95_per_image_ms": float(np.percentile(per_image, 95) * 1000),
        "throughput_fps": float(batch_size / np.median(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run)
    device = torch.device("cpu")
    model = build_model().to(device)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)

    n_threads_default = torch.get_num_threads()
    results = {
        "run": run_dir.name,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "torch_threads": n_threads_default,
        "model": "ResNet18 (frozen backbone) + linear head, 6 classes",
        "input_resolution": "3x224x224",
        "batches": [],
    }
    for bs in (1, 4, 16, 32):
        r = _bench(model, batch_size=bs, n_iters=args.n_iters, warmup=args.warmup)
        results["batches"].append(r)
        print(
            f"batch={bs:>2}  "
            f"p50={r['p50_batch_ms']:>7.2f}ms ({r['p50_per_image_ms']:>5.2f}ms/img)  "
            f"p95={r['p95_batch_ms']:>7.2f}ms  "
            f"throughput={r['throughput_fps']:>6.1f} FPS"
        )

    out_dir = ROOT / "reports" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latency.json").write_text(json.dumps(results, indent=2))

    md = ["# CPU latency benchmark", ""]
    md.append(f"- run: `{results['run']}`")
    md.append(f"- platform: {results['platform']}")
    md.append(f"- torch: {results['torch_version']}, intra-op threads: {results['torch_threads']}")
    md.append(f"- model: {results['model']}")
    md.append("")
    md.append("| batch | p50 (ms) | p95 (ms) | per-image p50 (ms) | throughput (FPS) |")
    md.append("|------:|---------:|---------:|-------------------:|-----------------:|")
    for r in results["batches"]:
        md.append(
            f"| {r['batch_size']:>5} | {r['p50_batch_ms']:8.2f} | {r['p95_batch_ms']:8.2f} | "
            f"{r['p50_per_image_ms']:18.2f} | {r['throughput_fps']:16.1f} |"
        )
    md.append("")
    md.append("Methodology: warmup of 5 forward passes, then 30 timed iterations per batch size.")
    md.append("Timing via `time.perf_counter()` around `model(x)` only — preprocessing is")
    md.append("not measured. p50 / p95 are over the 30-iteration sample.")
    (out_dir / "latency.md").write_text("\n".join(md))
    print(f"\nWrote {out_dir / 'latency.md'}")


if __name__ == "__main__":
    main()
