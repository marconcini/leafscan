# Research log

Dated entries, newest first. Each entry: **question → what we did → result → takeaway**.
Negative or surprising results are kept on purpose.

---

## 2026-10-15 — Baseline: EfficientNet-B0 reaches 99.65%, errors are biologically meaningful

**Question.** What does a light CNN (EfficientNet-B0, 5.3M params) reach on the leak-free split,
and how far is it from the ~99% usually reported on PlantVillage [Mohanty et al. 2016]?

**Hypothesis.** Very high (>97%): images are taken in controlled lab conditions, so the task is
easy.

**What we did.** ImageNet-pretrained EfficientNet-B0, full fine-tuning, 5 epochs, 224 px,
AdamW (lr 1e-3, warmup + cosine), label smoothing 0.1. 21 min on an Apple M-series GPU.
Config, git commit and library versions in `reports/efficientnet_b0/config.yaml`.

**Result.**
- Validation accuracy 99.0% after the *first* epoch, 99.8% after 5.
- Test: **99.65% accuracy, 99.46% macro-F1**, 38 errors / 10,709.
- Error analysis (`reports/efficientnet_b0/test_errors.json`):
  - 18/38 errors: corn *Cercospora / gray leaf spot* ↔ *Northern leaf blight*. Both produce
    elongated grey-brown lesions, and gray leaf spot has about half as many training
    images (365 vs 689).
  - 5/38: *potato* late blight predicted as *tomato* late blight — same pathogen
    (*Phytophthora infestans*), similar lesions. The model gets the disease right, the crop wrong.
  - Mean confidence: 0.89 on correct predictions vs 0.64 on errors (label smoothing caps
    confidence, hence 0.89 rather than ~1.0).

**Takeaway.** Hypothesis confirmed: on lab images the task is close to solved, and remaining
errors are mostly between visually similar diseases rather than random. A 99.65% score says
little about real-world use — this motivates the background-bias and field-photo experiments.

**New questions.**
- Is the gap between correct/wrong confidence enough to *abstain* on uncertain photos? → calibration step.
- Would the potato/tomato confusion disappear if the user told us the crop? (cheap, realistic
  for an app).
- Leakage effect still to quantify: same model trained on a naive random split.

---

## 2026-10-15 — Plan: baseline and how much leakage matters

**Question.** What does a light CNN (EfficientNet-B0, 5.3M params) reach on the leak-free split,
and how far is it from the ~99% usually reported on PlantVillage [Mohanty et al. 2016]?

**Hypothesis.** Still very high (>97%): PlantVillage images are taken in controlled lab
conditions, so the task is easy. The interesting questions are elsewhere (background bias,
field photos), which is why those are on the roadmap.

**Open question for later.** Quantify the effect of leakage: train the same model with a naive
random split and compare validation vs test accuracy for both splits.

---

## 2026-10-14 — Data leakage: the file name is not the leaf

**Question.** PlantVillage photographed each physical leaf several times. How do we keep all
photos of a leaf on the same side of the train/validation split?

**What we did.**
1. First attempt: group images by the identifier in the file name
   (`<uuid>___RS_HL 9155.JPG` → `rs_hl 9155`).
   Result: 99% of "groups" contained a single image → this ID identifies a *photo*, not a leaf.
   Grouping by it does nothing.
2. Used the dataset's `leaf-map.json` (photo ID → physical leaf). It covers 76% of images
   (33,078 / 43,596 train), with ~4 photos per leaf.
3. Grouping on the first leaf of the map still left **35 leaves shared** between train and
   validation. Cause: some photo IDs exist in several classes, and the map lists one leaf per
   class. Fix: pick the leaf whose class matches the image's folder.

**Result.** 0 leaves shared between train and validation (enforced by a unit test).
The official test split itself shares 5 leaves with train — an upstream limitation we keep
as-is to stay comparable with published results.

**Takeaway.** Don't trust what an identifier looks like; check group sizes. A one-line sanity
check (images per group) revealed the first approach was silently useless.
