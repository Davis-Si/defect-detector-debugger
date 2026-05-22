"""Train a frozen-backbone classifier on NEU-CLS.

CLI:
    python -m src.train --augment none --epochs 8 --run-name baseline
    python -m src.train --augment flip --epochs 8 --run-name flip
    python -m src.train --augment flip_rotate --epochs 8 --run-name flip_rotate

Outputs (under runs/<run-name>/):
    config.json           -- exact run configuration
    metrics.json          -- per-epoch train/val loss + accuracy
    model.pt              -- final model weights (head only is what changed)
    test_predictions.csv  -- one row per test image: idx, label, pred, conf, top-2-conf
    test_features.npy     -- 512-d features for every test image (for t-SNE etc.)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .data import CLASS_NAMES, ClassAwareDataset, NEUCLS, stratified_train_val_split
from .model import build_model, build_transform, extract_features, trainable_parameters

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device) -> tuple[float, float, list]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    rows: list[tuple[int, int, int, float, float]] = []
    with torch.no_grad():
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += loss.item()
            probs = F.softmax(logits, dim=1)
            top2 = probs.topk(2, dim=1)
            pred = top2.indices[:, 0]
            conf = top2.values[:, 0]
            second = top2.values[:, 1]
            correct += (pred == y).sum().item()
            total += y.numel()
            for i in range(y.numel()):
                rows.append(
                    (
                        int(idx[i]),
                        int(y[i]),
                        int(pred[i]),
                        float(conf[i]),
                        float(second[i]),
                    )
                )
    return loss_sum / total, correct / total, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augment",
        choices=["none", "flip", "flip_rotate", "flip_rotate_mild", "class_aware", "auto"],
        default="none",
        help=(
            "class_aware = hand-picked partition (skip rotation for inclusion+scratches). "
            "auto       = read reports/sensitivity/policy.json; rotate only the classes "
            "the empirical sensitivity probe found rotation-safe."
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    eval_tf = build_transform(train=False)

    if args.augment in ("class_aware", "auto"):
        if args.augment == "class_aware":
            # Hand-picked partition: see findings.md section 3c. Rotation
            # destroys class signal for `inclusion` and `scratches`, so those
            # get flip-only; the rest get flip + ±15° rotation.
            no_rotation = (
                CLASS_NAMES.index("inclusion"),
                CLASS_NAMES.index("scratches"),
            )
        else:  # auto
            policy_path = ROOT / "reports" / "sensitivity" / "policy.json"
            if not policy_path.exists():
                raise FileNotFoundError(
                    f"{policy_path} not found. Run `python -m src.sensitivity --run runs/baseline` first."
                )
            policy = json.loads(policy_path.read_text())
            # The probe's "sensitive" classes are the ones we should *not* rotate.
            no_rotation = tuple(CLASS_NAMES.index(c) for c in policy["sensitive_classes"])
            print(f"[auto] policy from {policy_path.name}:")
            print(f"  rotation-SAFE (will be rotated):     {policy['safe_classes']}")
            print(f"  rotation-SENSITIVE (flip only):       {policy['sensitive_classes']}")
        train_full_base = NEUCLS(DATA_DIR / "train.parquet", transform=None)
        train_full = ClassAwareDataset(
            train_full_base,
            transform_default=build_transform(train=True, augment="flip_rotate"),
            transform_no_rotation=build_transform(train=True, augment="flip"),
            no_rotation_classes=no_rotation,
        )
    else:
        train_tf = build_transform(train=True, augment=args.augment)
        train_full = NEUCLS(DATA_DIR / "train.parquet", transform=train_tf)

    train_full_eval = NEUCLS(DATA_DIR / "train.parquet", transform=eval_tf)
    test_set = NEUCLS(DATA_DIR / "test.parquet", transform=eval_tf)

    train_idx, val_idx = stratified_train_val_split(train_full.labels, val_frac=0.15, seed=args.seed)
    train_subset = Subset(train_full, train_idx.tolist())
    val_subset = Subset(train_full_eval, val_idx.tolist())

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model().to(device)
    optim = torch.optim.Adam(trainable_parameters(model), lr=args.lr)

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    history: list[dict] = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, count = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * y.numel()
            count += y.numel()
        train_loss = running / count
        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": epoch_time,
            }
        )
        print(
            f"[{args.run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({epoch_time:.1f}s)"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Load best-on-val and run final test eval.
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, rows = evaluate(model, test_loader, device)
    print(f"[{args.run_name}] TEST loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save artefacts.
    torch.save(model.state_dict(), run_dir / "model.pt")
    metrics = {
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    import csv

    with (run_dir / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "pred", "confidence", "second_confidence"])
        w.writerows(rows)

    # Extract features for the entire test set, in label order, for t-SNE etc.
    feats: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _, _ in DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ):
            feats.append(extract_features(model, x.to(device)).cpu().numpy())
    np.save(run_dir / "test_features.npy", np.concatenate(feats, axis=0))


if __name__ == "__main__":
    main()
