# Industrial Defect Classification — Failure Analysis

End-to-end failure analysis of a steel-surface-defect classifier on the
[NEU-CLS dataset](https://huggingface.co/datasets/newguyme/neu_cls)
(1,800 images, 6 defect classes). The interesting deliverable here is not the
model — a frozen ImageNet ResNet-18 with a linear head clears 98% test accuracy
in a few minutes on CPU — but the toolkit and report that systematically dissect
*where* and *why* the model fails, and quantify which "obvious" fixes actually
help.

## Headline results

Six training runs ablating different augmentation policies on a 1,440-image
training set. All errors below are on a held-out test set of 360 images.

| Run                | Augmentation policy                                              | Test accuracy |
|--------------------|------------------------------------------------------------------|---------------|
| **baseline**       | none                                                             | **0.9806**    |
| flip               | horizontal flip                                                  | 0.9722        |
| flip_rotate        | h-flip + ±15° rotation                                           | 0.9139        |
| auto_sensitivity   | h-flip + ±15° rotation, applied per-class via empirical probe    | 0.9083        |
| flip_rotate_mild   | h-flip + ±5° rotation                                            | 0.9028        |
| class_aware        | rotation skipped for hand-picked classes (inclusion + scratches) | 0.8611        |

![Ablation across augmentation regimes](reports/ablation.png)

**The headline finding is that every augmentation regime hurts on this
dataset.** The baseline (no augmentation) wins by 0.9 to 12 points depending
on the alternative. The story that connects the runs is the interesting
part — see [`findings.md`](findings.md) Section 3 for the full trace.

The `auto_sensitivity` run is novel: instead of guessing which classes are
"rotation-safe", `src/sensitivity.py` empirically probes the trained model
by measuring how much each class's predicted probability drops after
applying each candidate transform. The probe inverts what intuition
suggests:

| class            | sensitivity to ±15° rotation (drop in p_true) | I assumed         | Probe says        |
|------------------|----------------------------------------------:|-------------------|-------------------|
| scratches        | 0.032                                         | sensitive (skip)  | **safe** (rotate) |
| patches          | 0.105                                         | safe              | safe              |
| inclusion        | 0.399                                         | sensitive (skip)  | safe (rotate)     |
| crazing          | 0.478                                         | safe              | sensitive (skip)  |
| pitted_surface   | 0.516                                         | safe              | sensitive (skip)  |
| rolled-in_scale  | 0.603                                         | safe              | sensitive (skip)  |

The auto-derived policy beats the hand-picked `class_aware` policy by 4.7
points (0.9083 vs 0.8611), confirming the probe extracts real signal — but
still loses to the no-augmentation baseline by 7 points, which finally
proves the dataset just doesn't accept geometric augmentation under this
model. **A negative result that's robust because we ran the principled
experiment.**

![Per-class augmentation sensitivity heatmap](reports/sensitivity/per_class_sensitivity.png)

### Production-readiness summary

The baseline model isn't just accurate, it's *deployable*:

| Metric                                  | fp32                                | INT8 dynamic           |
|-----------------------------------------|-------------------------------------|------------------------|
| Test accuracy                           | 98.06%                              | **98.06% (no loss)**   |
| Confidence-thresholded auto-decision    | 100% accurate at T=0.70             | (same threshold reuses) |
| Auto-decided coverage at T=0.70         | **92.5%** (333/360 frames)          | —                      |
| Defer-to-human rate                     | 7.5%                                | —                      |
| **CPU latency p50 (batch=1)**           | 40.1 ms / frame → 25.0 FPS           | **26.3 ms / frame → 38.0 FPS (1.5× faster)** |
| CPU latency p50 (batch=32)              | 15.6 ms / frame → 64.1 FPS           | 15.4 ms / frame → 65.1 FPS |
| Hardware                                | 8-core CPU, no GPU                  | 8-core CPU, no GPU     |

INT8 dynamic quantization gives a **1.5× speedup at batch=1 with zero
accuracy loss**. For a streaming-frame factory deployment (the dominant
case — one image at a time as the line emits frames) that's the difference
between 25 and 38 FPS. Pareto plot, full table, and reproducer at
`reports/quantization/`.

![Accuracy vs latency Pareto: fp32 vs INT8 dynamic](reports/quantization/pareto.png)
![Deployment policy: coverage vs auto-decision accuracy](reports/deployment/coverage_accuracy.png)

### Failure-mode visuals

Confusion matrix and Grad-CAM saliency on the dominant confused class pair
(`inclusion` mistaken for `scratches`):

![Baseline confusion matrix](runs/baseline/analysis/confusion_matrix.png)
![Grad-CAM: inclusion predicted as scratches](runs/baseline/analysis/gradcam_confused_pairs/gradcam__inclusion__as__scratches.png)

The Grad-CAM heatmaps show the model is *not* getting distracted by background
artefacts — it's looking exactly at the elongated dark streaks, which on these
samples really do resemble scratches. So the residual error is a true visual
ambiguity, not a model attention problem. This is a deployment-relevant
distinction: it tells the team to invest in either an inclusion-vs-streak
auxiliary classifier or a relabelling pass on edge cases, *not* in retraining
the main model.

Calibration is reliable enough to use as the deployment signal:

![Baseline calibration](runs/baseline/analysis/calibration.png)
![t-SNE of baseline test features](runs/baseline/analysis/tsne.png)

## What's in the repo

```
src/
  data.py        NEU-CLS parquet loader, stratified split, ClassAwareDataset
  model.py       ResNet-18 backbone (frozen) + linear head, transforms
  train.py       Training loop with per-run config + artefact dump
  analyze.py     Failure-analysis toolkit (per-class, calibration, t-SNE, ...)
  gradcam.py     Grad-CAM saliency overlays for any run
  sensitivity.py Per-class augmentation sensitivity probe (auto-derives a policy)
  deployment.py  Coverage-vs-accuracy sweep + threshold recommendation
  benchmark.py   CPU latency benchmark across batch sizes
  quantize.py    INT8 dynamic quantization + accuracy/latency Pareto
runs/<name>/
  config.json, metrics.json, model.pt, test_predictions.csv, test_features.npy
  analysis/
    per_class.{csv,png}, confusion_matrix.png, calibration.png, tsne.png
    confused_pairs/                 image galleries by (true, pred) pair
    hard_examples/                  high-conf-wrong + low-conf-right boards
    gradcam_confused_pairs/         Grad-CAM overlays for confused pairs
    gradcam_high_conf_wrong.png     Grad-CAM on the highest-conf errors
    summary.json
reports/
  ablation.{csv,png}                cross-run accuracy comparison
  sensitivity/per_class_sensitivity.{csv,png}    per-class transform sensitivity
  sensitivity/policy.json                         auto-derived augmentation policy
  deployment/coverage_accuracy.{csv,png}
  deployment/recommendation.json    recommended confidence threshold + coverage
  benchmark/latency.{json,md}       CPU inference benchmark
  quantization/{pareto.png,benchmark.csv,summary.json}  fp32 vs INT8 dynamic
  summaries.json
findings.md                         narrative writeup of every failure mode
Makefile                            `make all` reproduces every artefact
requirements.txt
```

## Failure analysis toolkit

`src/analyze.py` produces the following per run:

1. **Per-class precision / recall / F1** with support — bar chart + CSV.
2. **Confusion matrix** as a heatmap.
3. **Confidence-binned accuracy** (calibration). The model is well-calibrated
   on the high-confidence regime (>0.7 confidence → 100% accurate, n=333) and
   uncertain on the long tail (<0.5 confidence → 62.5% accurate, n=8). All 7
   baseline errors have confidence under 0.7.
4. **Most-confused class-pair galleries.** For each of the top-3 off-diagonal
   confusion-matrix entries, dump up to 8 misclassified images sorted by model
   confidence — the fastest way to see whether a confusion is a *labelling
   problem* or a *real visual ambiguity*.
5. **Hard examples board.**
   - "High-confidence wrong": the model was sure but wrong (likely label noise
     or genuinely out-of-distribution samples).
   - "Low-confidence right": the model just barely got it (genuinely ambiguous
     samples worth showing the annotation team).
6. **t-SNE of penultimate features**, plotted twice: coloured by true class
   (cluster structure check) and by correctness (where do errors live in
   feature space?).
7. **Cross-run ablation table** — same metrics across all augmentation regimes,
   side by side.

`src/gradcam.py` adds Grad-CAM saliency overlays so you can see *which pixels*
the model used for any prediction. In practice this is what tells the
annotation team whether to relabel a sample or whether the visual signal is
genuinely ambiguous. Run with `python -m src.gradcam --run runs/baseline`.

`src/sensitivity.py` is the novel diagnostic. For each (class, transform)
cell it measures how much the trained model's predicted probability for the
true class drops when the transform is applied at test time. Output: a
heatmap of per-class sensitivity, plus a `policy.json` that auto-derives
which classes should be rotated at training time. The result on this
dataset directly contradicted my class-name-based guess (see headline) —
which is exactly why the diagnostic matters. Run with
`python -m src.sensitivity --run runs/baseline`.

`src/deployment.py` computes the coverage-vs-accuracy curve and recommends a
deployment threshold. Run with `python -m src.deployment --run runs/baseline`.

`src/benchmark.py` measures p50 / p95 inference latency on CPU across batch
sizes. `src/quantize.py` extends this to compare fp32 vs INT8 dynamic
quantization on both accuracy and latency, producing the Pareto plot.
Run with `python -m src.quantize --run runs/baseline`.

All six operate on the artefacts emitted by `train.py`, so they work on
*any* run without re-training. That separation is intentional: in practice
you spend more time analysing runs than producing them.

## Reproducing

```bash
make setup     # creates .venv, installs CPU torch + deps
make data      # downloads the parquet files (~70 MB)
make train     # 5 runs × 6 epochs ≈ 25 min on 8-core CPU
make analyze   # writes analysis/ subdirs and reports/
```

Or, equivalently, `make all`.

The ResNet-18 ImageNet weights (~45 MB) are downloaded by torchvision on the
first training run and cached under `~/.cache/torch/hub/`.

## Why this design

The job posting that motivated this project asks for someone who debugs CV
systems through "targeted evaluations rather than only looking at aggregate
metrics" and traces problems through "datasets, labels, augmentations, and
training behaviour". The repo is structured to make every one of those a
first-class concern:

- *Datasets / labels* — `confused_pairs/` and `hard_examples/` galleries make
  label noise visible in seconds.
- *Augmentations* — six runs, hypothesis-driven ablation including a
  principled hand-picked fix and an empirically-derived auto policy, *both
  of which failed*, written up in `findings.md` Sections 3 and 8.
- *Training behaviour* — `metrics.json` keeps per-epoch loss + accuracy and
  best-val checkpointing; `findings.md` discusses what the curves imply.
- *Targeted evaluation* — sliced confidence bins and per-class metrics force
  you to look past the aggregate number.

See [`findings.md`](findings.md) for the actual narrative.
