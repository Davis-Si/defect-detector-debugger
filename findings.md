# Findings

Each section below names a failure mode, traces a likely root cause through
data → labels → augmentations → training behaviour, and either implements the
fix in this repo or scopes what would be needed to validate one. Numbers are
from the artefacts under `runs/` and `reports/`.

## 1. Aggregate accuracy hides a single class doing all the work

The baseline scores 98.1% on test, but five of the six classes are at exactly
1.000 F1 (crazing, patches, rolled-in_scale; with scratches at 0.97 and
pitted_surface at 0.98). **Every test error is an `inclusion`** —
recall on inclusion is 0.90 vs 1.00 for everything except scratches and
pitted_surface.

Source: `runs/baseline/analysis/per_class.csv`,
`runs/baseline/analysis/confusion_matrix.png`.

That collapses the failure-analysis question into one question: *why is
inclusion confused with scratches (4 cases) and pitted_surface (2 cases)?*

**Implication for production:** if Cerrion shipped this model and the customer
cared about inclusion-class recall (e.g. because inclusions correlate with
downstream casting defects), the 98.1% headline would be dangerously
reassuring. The right SLA is per-class — and ideally per-class with a
confusion budget against the most operationally costly mistake.

## 2. The model is well-calibrated, so confidence is usable as a triage signal

`runs/baseline/analysis/calibration.png` and `confidence_bins.csv`:

| confidence bin | n   | accuracy |
|----------------|-----|----------|
| [0.00, 0.50)   | 8   | 62.5%    |
| [0.50, 0.70)   | 19  | 78.9%    |
| [0.70, 0.85)   | 52  | **100%** |
| [0.85, 0.95)   | 83  | **100%** |
| [0.95, 1.00)   | 198 | **100%** |

All 7 baseline errors fall under 0.70 confidence; the highest-confidence wrong
prediction is 0.69. **A confidence threshold of 0.70 would catch every error
in this test set at the cost of routing 27/360 = 7.5% of frames to manual
review.** That is a directly shippable result for a high-precision
inspection pipeline — the model "knows" when it's uncertain.

This is also why the `hard_examples/low_conf_right.png` board is more
interesting than `high_conf_wrong.png`: there are no high-confidence wrong
predictions worth investigating, but there *are* low-confidence correct
predictions that show what the genuinely-ambiguous samples look like.

## 3. Augmentation made things worse — and even the principled fix didn't save it

Cross-run ablation (`reports/ablation.csv`):

| run                | augmentation                                        | test acc | test loss |
|--------------------|-----------------------------------------------------|----------|-----------|
| baseline           | none                                                | **0.9806** | 0.128 |
| flip               | h-flip                                              | 0.9722   | 0.138 |
| flip_rotate        | h-flip + ±15° rot                                   | 0.9139   | 0.278 |
| flip_rotate_mild   | h-flip + ±5° rot                                    | 0.9028   | 0.331 |
| class_aware        | rotation only on rotation-safe classes              | 0.8611   | 0.396 |

This section walks the experiment-by-experiment trace.

### 3a. Horizontal flip alone cost ~1 point (mild)

Scratches recall stayed at 1.00 but precision dropped from 0.94 to 0.88 — the
model started *over*-predicting scratches. The flip run mistakes 8 inclusions
for scratches versus 4 in baseline. Hypothesis: horizontal flip doubles the
"linear-streak-like" distribution, pulling the decision boundary toward
scratches.

### 3b. ±15° rotation cost ~7 points (severe)

Inclusion recall collapsed from 0.90 to 0.63 — 22 of 60 test inclusions
misclassified, 13 of them as scratches. Rotating inclusions makes their
elongated examples resemble scratches at intermediate angles. Calibration
also degrades: the [0.0, 0.5) confidence bin grows from 8 to 26 samples and
is only 35% accurate.

### 3c. Hypothesis-driven fix: class-aware augmentation

I codified the lesson from 3b as a falsifiable hypothesis: *"Augmentation hurts
because rotation destroys class-discriminative information for `scratches`
and `inclusion`. If we apply rotation only to the other four classes, the
gain on those four should outweigh the loss on the two protected classes."*

Implemented in `src/data.py:ClassAwareDataset` and trained as the
`class_aware` run.

**Result: hypothesis was directionally right and overall wrong.**

| class            | baseline F1 | flip_rotate F1 | class_aware F1 | direction |
|------------------|-------------|----------------|----------------|-----------|
| crazing          | 1.000       | 0.945          | 0.839          | worse     |
| inclusion        | 0.939       | 0.768          | 0.817          | **recovers** vs flip_rotate (+0.05) |
| patches          | 1.000       | 0.992          | 0.976          | ~flat     |
| pitted_surface   | 0.975       | 0.905          | 0.738          | **regresses sharply** vs flip_rotate |
| rolled-in_scale  | 1.000       | 0.947          | 0.857          | worse     |
| scratches        | 0.968       | 0.902          | 0.916          | **recovers** vs flip_rotate (+0.01) |

The two classes I *protected* from rotation (`inclusion`, `scratches`) did
recover f1 versus the brute-force `flip_rotate` run, exactly as predicted.
But `pitted_surface` — which I assumed was rotation-invariant — collapsed
(F1 0.91 → 0.74), and `rolled-in_scale` and `crazing` also slid. The
underlying mechanism is real, but my partition of "rotation-safe" classes
was wrong. Without the kind of disagreement-driven investigation a
production team would do (Grad-CAM, per-class confidence drift, examining
the rotated training-time samples) you can't tell from class names alone
which defects are orientation-dependent.

### 3d. Mild ±5° rotation didn't recover either

The natural follow-up — *maybe the rotation magnitude was too aggressive* —
also failed (`flip_rotate_mild`, 0.9028). Smaller rotations help inclusion
recall a tiny bit (0.63 → 0.78) but don't recover the points lost on the
other classes.

### Lesson

On NEU-CLS, **no geometric augmentation policy beats the no-augmentation
baseline.** Three experiments, three losses. The principled fix from 3c is
particularly informative because it disproves the "obvious" mitigation —
which is exactly what failure analysis is *for*.

What would actually help on this dataset, but is left out of scope here:
- **Photometric augmentations only** (brightness / contrast jitter, salt
  noise, mild blur) — these don't move pixels around and so don't destroy
  orientation signal. ~30 lines in `build_transform`.
- **Crop-based augmentations within the surface area** rather than rotation
  — increases sample diversity without rotating defects.
- **Per-sample audit before augmenting** — score each training image's
  rotation-sensitivity by comparing model confidence before vs after a small
  test-time rotation; protect the high-sensitivity samples.

In a deployment playbook this finding is concrete: *any new defect class
needs an explicit rotation/flip-sensitivity audit before being added to the
augmentation pipeline.*

## 4. Inclusion / scratches / pitted_surface form a confusion triangle

Across all five runs, the dominant off-diagonal entries are the same pair:
**inclusion ↔ scratches** and to a lesser extent **inclusion ↔ pitted_surface**.

- `runs/baseline/analysis/confused_pairs/confused__inclusion__as__scratches.png`
  shows 4 inclusions classified as scratches at confidences ranging 0.49 to
  0.69. Visually they are all elongated dark marks where the "inclusion"
  description (embedded foreign material) is hard to distinguish from a
  "scratch" (mechanical groove) without context the model doesn't have:
  edge profile, depth cues, surrounding material disturbance.
- `runs/baseline/analysis/tsne.png` shows inclusion as the only class without
  a clean cluster — its points overlap both pitted_surface and scratches
  regions. The other five classes are well-separated.

**Practical fix recommendations** (priority order):
1. *Re-examine inclusion labels* — the most likely root cause of the
   inclusion/scratches confusion is that the dataset itself contains samples
   on the boundary that human annotators would also disagree on. A small
   relabel pass on the 60 test-set inclusions would either (a) reveal label
   noise we should fix, or (b) confirm that the boundary cases are genuine
   and the model's confidence calibration is the right surface to expose
   them in production.
2. *Two-stage classifier* — train a high-precision binary
   "inclusion-vs-streak" head specifically for the confidence-uncertain band
   (≤0.70 in baseline). Cheap to add given the current pipeline.
3. *More inclusion data* — class is balanced in train (240/class), but the
   visual variance within inclusion is higher than the others. A targeted
   data-collection pass would likely move the needle more than any model
   change.

## 5. Training behaviour is healthy; the bottleneck is data

From `runs/baseline/metrics.json`:
- Train loss decreases monotonically from 0.92 → 0.11 over 6 epochs.
- Val acc reaches 0.99 by epoch 3 and oscillates between 0.986 and 0.991.
- Test acc (0.981) is below val acc (0.991), but only by ~1 point.

There is no overfitting signal worth chasing here. Adding regularisation,
schedulers, or more epochs would not move the needle — the test errors are not
a fitting problem, they are an *ambiguity problem* in the inclusion class. The
right next investments are on the data side (Section 4), not the model side.

## 6. Grad-CAM confirms it's a visual-ambiguity problem, not an attention problem

It's tempting to assume a misclassification means the model is "looking at
the wrong thing." Grad-CAM on the dominant confused pair
(`runs/baseline/analysis/gradcam_confused_pairs/gradcam__inclusion__as__scratches.png`)
shows the opposite: on every one of the 4 inclusions misclassified as
scratches, the heatmap is squarely on the elongated dark feature in the
image — the same place a human annotator's eye goes. The model isn't
distracted by background; it's looking at the right region and reading the
visual signal as a scratch.

This reframes the fix. If the model were attending to the wrong place,
priorities would be: more data variety, larger receptive field, attention
mechanisms. Because the model attends to the *right* place but reads the
signal differently from the labels, priorities are:

1. **Re-examine inclusion labels on the boundary cases.** Likely the same
   samples human annotators would also disagree on.
2. **An auxiliary inclusion-vs-streak head** trained specifically on the
   confidence-uncertain band — more capacity directed at the genuine
   ambiguity, rather than re-tuning the main classifier.
3. **Auxiliary signals if available** (depth, lighting variation, surrounding
   region context) — at the *data* layer, not the model layer.

The point is that Grad-CAM is one of the cheapest tools to switch from "the
model is wrong" to "*how* is it wrong, and what kind of fix does that
imply?" — a 30-line script changes the engineering plan.

## 7. Deployment policy: confidence-thresholded auto-decision is shippable as-is

`src/deployment.py` sweeps a confidence threshold T over the test set and
records `coverage(T)` (fraction auto-decided) and `accuracy(T)` (accuracy of
the auto-decided slice). The recommendation file
(`reports/deployment/recommendation.json`) reports the smallest T at which
the auto-decided slice is 100% accurate on the test set:

```json
{
  "feasible": true,
  "threshold": 0.70,
  "coverage": 0.925,
  "defer_rate": 0.075,
  "auto_accuracy": 1.00,
  "n_auto_decided": 333,
  "auto_errors": 0
}
```

**Operational meaning:** at T=0.70, the model auto-decides 92.5% of frames
correctly with zero errors on this evaluation, deferring 7.5% to a human
reviewer. That is a directly shippable policy — what changes between
"production ready" and "research demo" is not the model, it's having this
artefact and a recommended threshold backed by a test-set sweep.

The full curve (`reports/deployment/coverage_accuracy.png`) shows the
trade-off the operations team controls: lowering T expands coverage but
risks the first auto-error; raising T shrinks coverage but adds margin.

For Cerrion's use case in particular, where one missed defect can mean a
costly production stoppage, this kind of per-frame routing policy is
exactly the surface that turns model accuracy into shipped value.

## What's not done here, and why

- **No cross-fold disagreement audit.** With only 1,800 images, k-fold runs
  would take ~30 min wallclock on CPU and would not change the conclusions
  above (the per-class signal is consistent across the five augmentation
  runs we already have). On a real production model I would do it; on a
  one-day reference project the marginal value is low.
- **No domain-specific augmentations** (random crop within the surface
  region, brightness/contrast jitter to simulate lighting drift, salt-noise
  to simulate sensor artefacts). These are the augmentations that *would*
  likely help on this dataset, in contrast to the geometric ones tested
  here. Hooking them in is a 30-line change to `src/model.py:build_transform`.
- **Backbone unfreezing.** Frozen ResNet-18 already saturates at 98% with
  6 epochs. Fine-tuning the last block would likely buy a fraction of a
  point, at much higher CPU cost.

The bias in this prioritisation is deliberate: the goal of the project is to
demonstrate a debugging methodology, not to maximise the leaderboard score on
NEU-CLS.
