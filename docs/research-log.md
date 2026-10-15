# Research log

Dated entries, newest first. Each entry: **question → what we did → result → takeaway**.
Negative or surprising results are kept on purpose.

---

## 2026-10-15 — Next: baseline and how much leakage matters

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
