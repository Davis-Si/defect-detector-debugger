"""Grad-CAM for the ResNet-18 defect classifier.

Why this is here: the toolkit's confused-pair galleries answer "which images
does the model get wrong?". Grad-CAM answers the next question — "which
*pixels* did the model use?". For an industrial-defect classifier this is the
single most useful diagnostic, because it tells you whether a misclassification
is the model latching onto the wrong region (background, edge artefact,
specular highlight) or actually disagreeing on the defect itself.

Usage:
    python -m src.gradcam --run runs/baseline --mode confused_pairs --top-k 3
    python -m src.gradcam --run runs/baseline --mode high_conf_wrong
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .data import CLASS_NAMES, NEUCLS
from .model import IMAGENET_MEAN, IMAGENET_STD, build_model, build_transform

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class GradCAM:
    """Standard Grad-CAM. Hooks the last conv block of a ResNet-18.

    For ResNet-18 we use `layer4` (the 7x7 spatial map after the last residual
    stage) as is conventional. The activations and gradients are captured via
    forward / backward hooks during a single forward+backward pass.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: 1x3xHxW tensor (already normalized). Returns HxW heatmap in [0,1]."""
        self.model.zero_grad()
        # Frozen backbone has requires_grad=False on all params; we still need
        # the activations to carry gradients, so we re-enable grad on the
        # input path by toggling target features after the forward pass. The
        # easiest reliable approach is to temporarily flip requires_grad on
        # the layer4 parameters for the duration of the call.
        layer4_params = list(self.model.layer4.parameters())
        prev = [p.requires_grad for p in layer4_params]
        for p in layer4_params:
            p.requires_grad_(True)
        try:
            logits = self.model(x)
            score = logits[0, class_idx]
            score.backward()
        finally:
            for p, q in zip(layer4_params, prev):
                p.requires_grad_(q)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay(heatmap: np.ndarray, img_pil: Image.Image, alpha: float = 0.45) -> np.ndarray:
    """Render heatmap on top of grayscale source. Returns HxWx3 uint8."""
    img = np.asarray(img_pil.convert("RGB").resize(heatmap.shape[::-1])).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    heat = cmap(heatmap)[..., :3]  # drop alpha channel
    blended = (1 - alpha) * img + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _load_model(run_dir: Path, device: torch.device) -> torch.nn.Module:
    model = build_model().to(device)
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run `make train` to produce model checkpoints."
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def _gallery(rows: pd.DataFrame, model, cam, transform, test_pil, out_png, suptitle):
    if len(rows) == 0:
        return
    n = len(rows)
    cols = min(4, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols * 2, figsize=(cols * 4.4, rows_n * 2.6))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows.itertuples()):
        idx = int(r.index)
        img_pil = test_pil.get_pil(idx)
        x = transform(img_pil).unsqueeze(0)
        # Saliency for the model's *predicted* class — that's "what did the model see?".
        heat = cam(x, class_idx=int(r.pred))
        ov = overlay(heat, img_pil)
        ax_orig = axes[i // cols, (i % cols) * 2]
        ax_cam = axes[i // cols, (i % cols) * 2 + 1]
        ax_orig.imshow(img_pil, cmap="gray")
        ax_orig.set_title(
            f"true: {CLASS_NAMES[int(r.label)]}", fontsize=8
        )
        ax_orig.axis("off")
        ax_cam.imshow(ov)
        ax_cam.set_title(
            f"pred: {CLASS_NAMES[int(r.pred)]} ({r.confidence:.2f})", fontsize=8
        )
        ax_cam.axis("off")
    # Hide any unused axes.
    for j in range(n, rows_n * cols):
        axes[j // cols, (j % cols) * 2].axis("off")
        axes[j // cols, (j % cols) * 2 + 1].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def gradcam_confused_pairs(run_dir: Path, top_k: int = 3) -> None:
    device = torch.device("cpu")
    model = _load_model(run_dir, device)
    cam = GradCAM(model, model.layer4)
    transform = build_transform(train=False)
    test_pil = NEUCLS(DATA_DIR / "test.parquet", transform=None)

    preds = pd.read_csv(run_dir / "test_predictions.csv")
    out_dir = run_dir / "analysis" / "gradcam_confused_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same top-K (true,pred) pairs as the confused_pairs galleries.
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(preds["label"], preds["pred"], labels=list(range(len(CLASS_NAMES))))
    off = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                off.append((cm[i, j], i, j))
    off.sort(reverse=True)
    for count, true_c, pred_c in off[:top_k]:
        rows = (
            preds[(preds["label"] == true_c) & (preds["pred"] == pred_c)]
            .sort_values("confidence", ascending=False)
            .head(8)
        )
        suptitle = (
            f"Grad-CAM: True={CLASS_NAMES[true_c]}  Predicted={CLASS_NAMES[pred_c]}  "
            f"(n={count}, heatmap = where the model looked to predict {CLASS_NAMES[pred_c]})"
        )
        out = out_dir / f"gradcam__{CLASS_NAMES[true_c]}__as__{CLASS_NAMES[pred_c]}.png"
        _gallery(rows, model, cam, transform, test_pil, out, suptitle)
        print(f"  saved {out.name}")


def gradcam_high_conf_wrong(run_dir: Path) -> None:
    device = torch.device("cpu")
    model = _load_model(run_dir, device)
    cam = GradCAM(model, model.layer4)
    transform = build_transform(train=False)
    test_pil = NEUCLS(DATA_DIR / "test.parquet", transform=None)

    preds = pd.read_csv(run_dir / "test_predictions.csv")
    rows = (
        preds[preds["label"] != preds["pred"]]
        .sort_values("confidence", ascending=False)
        .head(8)
    )
    out = run_dir / "analysis" / "gradcam_high_conf_wrong.png"
    _gallery(
        rows,
        model,
        cam,
        transform,
        test_pil,
        out,
        "Grad-CAM on high-confidence wrong predictions",
    )
    print(f"  saved {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=["confused_pairs", "high_conf_wrong", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run_dir = Path(args.run)
    if args.mode in ("confused_pairs", "all"):
        gradcam_confused_pairs(run_dir, top_k=args.top_k)
    if args.mode in ("high_conf_wrong", "all"):
        gradcam_high_conf_wrong(run_dir)


if __name__ == "__main__":
    main()
