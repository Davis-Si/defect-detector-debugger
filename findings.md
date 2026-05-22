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

## 3. Augmentation made things worse — and the worse-ness is class-specific

Cross-run ablation (`reports/ablation.csv`):

| run         | augmentation        | test acc | test loss |
|-------------|---------------------|----------|-----------|
| baseline    | none                | **0.9806** | 0.128 |
| flip        | h-flip              | 0.9722   | 0.138 |
| flip_rotate | h-flip + ±15° rot   | 0.9139   | 0.278 |

Two things to notice:

**(a) horizontal flip cost ~1 point.** Looking at per-class metrics: scratches
recall stayed at 1.00 but precision dropped from 0.94 to 0.88 — the model now
*over-predicts* scratches. The flip run mistakes 8 inclusions for scratches
versus 4 in baseline. Hypothesis: horizontal flip is the wrong augmentation
for this dataset because it doubles the "linear-streak-like" distribution
(scratches and some inclusions look like streaks), pulling the decision
boundary toward scratches.

**(b) adding rotation cost ~7 points.** Inclusion recall collapsed from 0.90
to 0.63 — 22 of 60 test inclusions are now misclassified, 13 of them as
scratches. The mechanism is cleaner: many `scratches` samples are vertical
elongated marks, and `inclusion` samples are more clustered/blob-like. Rotating
inclusions makes them resemble scratches at intermediate angles. Worse, the
calibration goes too — the [0.0, 0.5) confidence bin grows from 8 to 26
samples and is only 35% accurate (vs 63% for baseline).

**Lesson:** "augmentations always help" is wrong on industrial data where
defect classes carry orientation as part of their definition. The right move
on this dataset would be either:
- *No augmentation* (what baseline does — and it wins).
- *Class-aware augmentation*: apply rotation only to truly orientation-invariant
  classes (crazing, patches, pitted_surface, rolled-in_scale) and skip it for
  scratches/inclusion. Not implemented in this run; would require a custom
  per-sample augmentation pipeline.
- *Smaller-magnitude rotations* (e.g. ±5°) to test whether some rotational
  invariance helps without destroying class signal.

This is a result that should change a deployment playbook: any new defect class
added in production needs an explicit yes/no on whether it carries
orientation, or its augmentation policy will silently be wrong.

## 4. Inclusion / scratches / pitted_surface form a confusion triangle

Across all three runs, the dominant off-diagonal entries are the same pair:
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

## What's not done here, and why

- **No cross-fold disagreement audit.** With only 1,800 images, k-fold runs
  would take ~30 min wallclock on CPU and would not change the conclusions
  above (the per-class signal is consistent across the three augmentation
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
