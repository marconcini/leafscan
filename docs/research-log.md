# Research log

Dated entries, newest first. Each entry: **question → what we did → result → takeaway**.
Negative or surprising results are kept on purpose.

---

## 2026-10-19 — Model comparison: architecture barely matters, FLOPs lie, the data has errors

**Results** (test split, 10,709 images):

| Model | Accuracy | Macro-F1 | Errors | Params | GFLOPs | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| ViT-S/16 | 99.73% | 99.59% | 29 | 21.7M | 8.5 | 29 ms |
| ConvNeXt-T | 99.71% | 99.55% | 31 | 27.8M | 8.9 | 435 ms |
| EfficientNet-B0 | 99.65% | 99.46% | 38 | 4.1M | 0.8 | 291 ms |
| ResNet-50 | 99.44% | 99.20% | 60 | 23.6M | 8.2 | 25 ms |

**Hypotheses vs. results.**
1. *All within ~0.3 points* → **confirmed** (0.29). Exact McNemar tests [Dietterich 1998]:
   ViT-S vs ConvNeXt-T p = 0.86, vs EfficientNet-B0 p = 0.19, ConvNeXt-T vs EfficientNet-B0
   p = 0.26 — no evidence of a difference. ResNet-50 is significantly worse than all three
   (p = 0.0038, 0.0001, 0.0002; Bonferroni threshold for 6 tests: 0.0083).
2. *21k/22k-pretrained models slightly ahead* → **consistent but not significant**: they are
   the top two, but not distinguishable from EfficientNet-B0. Pretraining data and architecture
   are confounded here.
3. *EfficientNet-B0 has the best accuracy/CPU-latency trade-off* → **refuted.** It is 11×
   slower than ResNet-50 on CPU despite 10× fewer FLOPs.

**Why FLOPs lied.** `torch.backends.mkldnn.is_available()` is False in this PyTorch macOS
build. Isolated test: a depthwise 7×7 conv (0.03 GFLOP) takes 5.8 ms, a dense 3×3 conv
(0.58 GFLOP) 0.9 ms. EfficientNet and ConvNeXt are built on depthwise convolutions, ViT and
ResNet on dense matmuls/convs that are well optimised. Latency depends on the kernels the
runtime has, not just on FLOPs. Open question: same measurement on x86 with oneDNN, or with
ONNX Runtime (the demo will probably run on such a server).

**Do models make the same errors?** No. Pairwise Jaccard overlap of error sets: 0.17–0.41.
94 test images fool at least one model, only 8 fool all four (`reports/wrong_for_all_models.jpg`):
- 1 photo of rotten **pepper fruits** labelled *Pepper bell / healthy* (a leaf class) → label error.
- 1 image of an **empty grey background** labelled *Soybean / healthy* → no leaf at all.
- 5 corn leaves (gray leaf spot ↔ northern leaf blight; one with no visible lesion).
- 1 apple scab leaf with barely visible symptoms, predicted healthy.

**Takeaways.**
- On PlantVillage, pick the model on cost, not on accuracy: differences are within noise.
- Measure latency on the target hardware; FLOPs are a poor proxy.
- Different errors → an ensemble might help (new roadmap item).
- At 99.7%, part of the remaining "errors" are the dataset's. Next: look for mislabelled and
  empty images systematically (new roadmap item).

**Engineering note.** The first ViT run slowed down ~30× (step 100 after 21 min) while the
Mac was idle; in isolation ViT-S trains as fast as EfficientNet-B0. Rerun under `caffeinate`
at normal speed. ConvNeXt-T epochs (~10 min) may have been affected too, so training times are
not compared.

---

## 2026-10-19 — Plan: does model choice matter on a near-solved dataset?

*Written before the runs finished, so the hypothesis can't be adjusted after the fact.*

**Question.** EfficientNet-B0 already reaches 99.65%. Do bigger or more recent architectures —
ResNet-50 (CNN, 2016), ConvNeXt-T (modern CNN, 2022, ImageNet-22k pretraining), ViT-S/16
(Transformer, ImageNet-21k pretraining) — bring anything, and at what cost?

**Setup.** Same data, splits, augmentations, epochs (5), batch size (64), schedule and seed.
Only the learning rate differs (usual fine-tuning value per architecture). No per-model
hyperparameter search: this compares architectures under one reasonable recipe, not each
architecture at its best.

**Hypotheses.**
1. All four models land within ~0.3 points of each other: the dataset is too easy to separate them.
2. Models pretrained on ImageNet-21k/22k (ConvNeXt-T, ViT-S) are slightly ahead.
3. EfficientNet-B0 offers the best accuracy / CPU-latency trade-off, which matters for a demo
   hosted on a CPU-only server.

**How we'll judge.** Accuracy and macro-F1 on the official test split, number of errors, and
whether models make *the same* errors (if they all fail on the same images, the problem is the
data, not the architecture). Cost: params, FLOPs, CPU latency (batch 1), GPU throughput.

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
