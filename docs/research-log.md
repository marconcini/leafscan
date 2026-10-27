# Research log

Dated entries, newest first. Each entry: **question → what we did → result → takeaway**.
Negative or surprising results are kept on purpose.

---

## 2026-10-27 — Plan: can the models say "I don't know"?

*Written before running the analysis.*

**Questions.** (1) Is confidence calibrated, in the lab and in the field? (2) If the model abstains
on uncertain photos, how much does field accuracy improve, and does a threshold chosen in the
lab still work in the field? (3) Can it tell that a photo is not a leaf at all?

**Setup.** The four PlantVillage models, no retraining.
- Calibration: expected calibration error (ECE, 15 bins) [Guo et al. 2017] on PlantVillage test
  and PlantDoc; temperature scaling fitted on the PlantVillage *validation* split.
- Selective prediction [Geifman & El-Yaniv 2017]: risk–coverage curves (answer only when
  confidence ≥ threshold). Lab threshold = lowest confidence keeping ≥ 99.9% accuracy on
  PlantVillage validation; applied unchanged to PlantDoc.
- Out-of-distribution: maximum softmax probability [Hendrycks & Gimpel 2017] and energy score
  [Liu et al. 2020]; AUROC for telling leaves from **flowers** (Oxford Flowers-102, near-OOD:
  plants but not leaves) and **textures** (DTD, far-OOD). Neither dataset is redistributed.

**Hypotheses.**
1. In the lab, models are **under**confident because of label smoothing (mean confidence ≈ 0.89
   vs. 99.6% accuracy); temperature scaling fixes it (ECE < 1%).
2. On PlantDoc they are **over**confident (mean confidence ≈ 0.5 vs. ~30% accuracy), and the
   lab-fitted temperature, which *sharpens* predictions, makes field calibration **worse**.
3. Abstention helps in the field: answering only on the 50% most confident PlantDoc photos
   raises accuracy from ~30% to ≥ 45%.
4. Lab thresholds do not transfer: the 99.9%-in-the-lab threshold answers few field photos, and
   far less than 99.9% of those answers are right.
5. Textures are easy to reject (AUROC > 0.9 against lab leaves), flowers harder (< 0.8).

---

## 2026-10-25 — Closing the gap: 233 field images are worth more than 39,000 lab images

**Results** (ConvNeXt-T, PlantDoc test split of 228 images, mean ± std over 3 seeds; 95%
bootstrap CI of a single run ≈ ±6 points):

| PlantDoc training images | From PlantVillage model | From ImageNet | PV − ImageNet, per seed |
|---:|---:|---:|---|
| 0 (zero-shot) | 38.6% | – | |
| 233 (10%) | **59.9 ± 1.8%** | 51.9 ± 2.0% | +7.9, +4.4, +11.8 |
| 581 (25%) | 63.9 ± 1.1% | 63.9 ± 0.7% | +0.9, +0.9, −1.8 |
| 1,161 (50%) | 69.7 ± 1.2% | 70.2 ± 0.4% | −2.2, +0.4, +0.4 |
| 2,323 (100%) | 74.4 ± 1.5% | **75.7 ± 2.9%** | −1.3, −3.1, +0.4 |

`reports/field_finetune/summary.png`. Same subset of images for both initialisations at a given
seed, so differences are paired.

**Hypotheses vs. results.**
1. *Full fine-tuning lifts accuracy to 55–70%* → **confirmed, and exceeded**: 74–76%. Health
   (healthy vs diseased) goes from ~80% zero-shot to 95%; the "default classes" disappear
   (most predicted class: 13 of 228 images).
2. *PlantVillage init beats ImageNet, mostly with little data* → **confirmed, but only with very
   little data.** With 233 images it is ahead on all 3 seeds (+8 points on average). From 581
   images on, the sign of the difference changes from seed to seed: no measurable benefit.
3. *Still rising at 100%* → **confirmed**: +5 points from 1,161 to 2,323 images, roughly
   linear in log(n). More field data would still help.

**Takeaways.**
- **Data that looks like the deployment beats a lot of data that doesn't.** 233 PlantDoc images
  (+21 points over zero-shot) do more than 39,251 PlantVillage images, which give 38.6%.
- Lab training is a useful head start only when field data is very scarce (a few hundred images).
- Each doubling of field data still adds ~5 points: the bottleneck is now the size of the field
  dataset, not the model.
- Caveat: one architecture, 228 test images, 3 seeds. Differences under ~3 points are noise.

---

## 2026-10-25 — Plan: closing the lab → field gap with a little field data

*Written before running the experiment.*

**Question.** Zero-shot, ConvNeXt-T gets 38.6% on PlantDoc's test split. How much does
fine-tuning on PlantDoc's own training images help, how many are needed, and does our
PlantVillage training still help compared with starting from ImageNet?

**Setup.** ConvNeXt-T, two initialisations: **PlantVillage** (our checkpoint, 99.7% lab) and
**ImageNet** (`convnext_tiny.fb_in22k_ft_in1k`, the weights PlantVillage training started from). Fine-tuned on
10 / 25 / 50 / 100% of PlantDoc train (2,323 images after cleaning, stratified subsets), same
budget of 400 optimiser steps for every fraction, 3 seeds each. Evaluated on PlantDoc's test
split (228 images), last checkpoint (no model selection on test). 95% bootstrap confidence
intervals: with 228 images, ±6 points is expected.

**Hypotheses.**
1. Full fine-tuning lifts test accuracy far above zero-shot: to 55–70%.
2. PlantVillage initialisation beats ImageNet initialisation, mostly with little data
   (10–25%); the gap shrinks at 100%.
3. Accuracy is still rising at 100%: the field dataset is too small to saturate.

---

## 2026-10-25 — Lab → field: from 99.7% to 31%

**Data actually used.** 2,551 PlantDoc images after cleaning. PlantDoc contained byte-identical
duplicates: 3 with the same label (kept once) and **9 groups (18 files) filed under conflicting
classes** — e.g. the same photo as both *potato early blight* and *potato late blight* (5 groups),
or under two corn diseases (3) — all dropped. 11 duplicates straddled PlantDoc's own
train/test split. Also: on a case-insensitive file system (macOS), `git clone` silently keeps one
of each pair of names differing only by case (`CAR1.jpg` / `car1.jpg`), 6 files.
PlantDoc images are **web-scraped**: real field photos, but also stock photos on white
backgrounds with watermarks. "Field" here means "not PlantVillage", not only "taken outdoors".

| Model | Lab (PlantVillage) | PlantDoc acc. | Balanced | Crop right | Healthy/diseased right | Conf. correct / wrong |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-T | 99.71% | **31.2%** | 29.7% | **58.8%** | 79.7% | 0.67 / 0.50 |
| ViT-S/16 | 99.73% | 29.7% | 28.3% | 51.4% | **80.0%** | 0.67 / 0.49 |
| ResNet-50 | 99.44% | 27.0% | 24.9% | 47.6% | 78.4% | 0.62 / 0.45 |
| EfficientNet-B0 | 99.65% | 23.4% | 22.5% | 44.9% | 75.7% | 0.57 / 0.43 |

Chance: 2.6% (38 classes). Healthy/diseased majority baseline: 66.8% (33.2% of images are healthy).
On PlantDoc's official test split only (228 images): 33–42%.

**Hypotheses vs. results.**
1. *Accuracy falls to 20–45%* → **confirmed**: 23–31%, in line with the 31% of
   [Mohanty et al. 2016] on images from other sources. The 99.7% was a lab number.
2. *ViT-S best, EfficientNet-B0 worst* → **partly confirmed.** EfficientNet-B0 is worst (as the
   background test predicted). ConvNeXt-T and ViT-S are tied (McNemar p = 0.08) and both
   significantly better than ResNet-50 and EfficientNet-B0 (p ≤ 0.003).
3. *Crop accuracy much higher than disease accuracy* → **confirmed**, but crop accuracy itself
   is only 45–59%: half the time the model does not even get the plant species right.
   Healthy/diseased is 76–80%, only 9–13 points above always answering "diseased".
4. *Overconfident on wrong field predictions (> 0.5)* → **refuted.** Mean confidence on errors
   is 0.43–0.50, and only 2–4% of errors have confidence > 0.9. Confidence on correct field
   predictions also drops (0.57–0.67, vs ~0.9 in the lab). Confidence carries signal on
   field photos → worth exploiting for abstention (calibration step).

**Where it breaks (ConvNeXt-T, classes with ≥ 30 images).**
- Collapsed classes: tomato mosaic virus 0%, **tomato bacterial spot 1% (100% in the lab)**,
  cherry healthy 2%, peach healthy 3%.
- Best: tomato late blight 79%, corn gray leaf spot 75%, squash powdery mildew 72%.
- **Default classes again**: ConvNeXt-T answers *tomato late blight* for 25% of PlantDoc images —
  the same class it answered for blank grey images in the background test. *Strawberry leaf
  scorch* plays the same role for all models. Unfamiliar input → a few "sink" classes.
- Top confusions are between diseases of the same crop (corn northern leaf blight / common rust →
  gray leaf spot; tomato septoria → early blight; leaf mold → late blight) and potato late blight
  → tomato late blight, as in the lab.

**Takeaways.**
- A PlantVillage model should not be presented as a plant-disease detector for real photos.
  This is the single most important number of the project so far.
- What predicted field robustness: the background-*change* sensitivity test (EfficientNet-B0
  worst in both), not lab accuracy (all four within 0.3 points) nor Grad-CAM leaf focus
  (ResNet-50 highest, yet third here).
- Next: (a) fine-tune on PlantDoc's train split and evaluate on its test split — how much does a
  little in-domain data help?; (b) use confidence to abstain; (c) data quality of PlantDoc itself.

---

## 2026-10-25 — Plan: lab → field, how much do the models drop?

*Written before running the evaluation.*

**Question.** Our models reach 99.4–99.7% on PlantVillage lab photos. What happens on real
photos taken in the field (cluttered backgrounds, several leaves, varied light)?

**Data.** PlantDoc [Singh et al. 2020], CC BY 4.0: ~2,500 web/field images, 28 classes, all of
which map to a PlantVillage class (e.g. *Tomato leaf yellow virus* → *Tomato Yellow Leaf Curl
Virus*, *Corn leaf blight* → *Northern Leaf Blight*). Our models never saw any PlantDoc image,
so **all** of them (train + test folders) are used for this zero-shot evaluation; the official
PlantDoc test split is kept for a later fine-tuning experiment.

**Metrics.** Prediction over all 38 PlantVillage classes (a prediction of an unmapped class,
e.g. *Orange / citrus greening*, is an error). Accuracy, balanced accuracy, and two coarser
levels: **crop** (right plant species?) and **health** (healthy vs. diseased right?). Mean
confidence on correct vs. wrong predictions.

**Hypotheses.**
1. **Large drop**: accuracy falls to 20–45%. [Mohanty et al. 2016] reported 31% when testing
   on images from other sources.
2. Ranking follows the background-robustness test: **ViT-S best, EfficientNet-B0 worst**.
3. Crop accuracy is much higher than disease accuracy (species is easier than disease).
4. Models stay **overconfident**: mean confidence on wrong field predictions > 0.5.

---

## 2026-10-21 — Background intervention: models don't use the background; Grad-CAM pointed at the wrong suspects

**Setup.** 10,157 test images (leaf mask available, leaf ≤ 90% of the frame). Accuracy /
**balanced accuracy** (mean of per-class accuracies, see pitfalls below). Chance = 2.6%.

| Model | Original | B. Leaf only | C. Background only (silhouette visible) | C′. Background only (box) | D. Swap: follows leaf / background / neither |
|---|---:|---:|---:|---:|---:|
| ViT-S/16 | 99.7 / 99.6 | 97.5 / **97.7** | 20.3 / 15.9 | 0.7 / 3.3 | 96.6 / 0.27 / 3.1 |
| ResNet-50 | 99.4 / 99.1 | 97.0 / 95.8 | 9.0 / 6.3 | 10.8 / 3.0 | 97.6 / 0.15 / 2.2 |
| ConvNeXt-T | 99.7 / 99.5 | 98.0 / 94.6 | 14.3 / 13.9 | 5.7 / 4.8 | 97.1 / 0.39 / 2.5 |
| EfficientNet-B0 | 99.6 / 99.4 | 89.6 / **84.5** | 15.5 / 12.6 | 2.3 / 1.4 | 91.6 / 1.25 / 7.2 |

**Hypotheses vs. results.**
1. *Background-only accuracy well above chance* → **refuted** once two artefacts are removed
   (C′, balanced): 1.4–4.8%, i.e. around chance. Background information exists in PlantVillage
   [Noyan 2022], but models trained on full images barely exploit it.
2. *ConvNeXt-T and ViT-S are the most background-dependent* → **refuted.** ViT-S is the *most*
   robust to background removal; EfficientNet-B0, the least (−15 points balanced). When the
   background is swapped, every model follows the leaf (≥ 91.6%) and almost never the
   background (≤ 1.25%).
3. *ResNet-50 keeps > 95% on leaf-only images* → **confirmed** (97.0%, 95.8% balanced).
4. *Tomato bacterial spot is among ConvNeXt-T's most background-dependent classes* →
   **refuted**: 100% → 97.2% with the leaf alone.

**Two artefacts that nearly produced the opposite conclusion.**
- **Leaf silhouette.** In C, the grey hole has the shape of the leaf. Hiding the whole bounding
  box instead (C′) makes ViT-S drop from 20.3% to 0.7%: what looked like "using the
  background" was recognising the *leaf shape*. This variant was added after looking at the
  first figure (`reports/background_conditions.jpg`), before running any model.
- **Default class.** On mostly-grey images, each model answers one class 86–96% of the time
  (ResNet-50: TYLCV). TYLCV is 10.9% of the images, so ResNet-50 scored 10.8% "accuracy" with
  no information at all. Balanced accuracy is immune to this: 3.0%.

**What the leaf-only drop means.** EfficientNet-B0 loses 10 points with the leaf alone, but
in the swap test it follows the new background only 1.25% of the time; its errors are mostly
"neither" (7.2%). So it is not *using* the original background: it is *disturbed* by an
unnatural grey one. That is a robustness issue (distribution shift), not a shortcut.

**Takeaways.**
- Grad-CAM attention on the background did **not** mean the background was used. The
  2026-10-20 suspicion about ConvNeXt-T and ViT-S does not survive a causal test: correlation
  (where the map is hot) ≠ causation (what changes the prediction).
- Interventions need their own sanity checks: look at the edited images, and use a metric that
  a constant answer cannot game.
- New lead for the lab → field step: sensitivity to background *changes* differs a lot between
  models (ViT-S most robust, EfficientNet-B0 least). Field photos have cluttered backgrounds,
  so this ranking is a prediction to test there.

---

## 2026-10-21 — Plan: do the models *use* the background? (causal test)

*Written before running the experiment.*

**Question.** Grad-CAM showed ConvNeXt-T and ViT-S spend 26–31% of their attention on the
background. Attention is not use. Does changing the background change the prediction?

**Method.** Test images with a leaf mask and leaf area ≤ 90% (same subset as Grad-CAM).
Pixels are replaced with the ImageNet mean colour (a neutral grey, i.e. zero after
normalisation). Four conditions, same trained models, no retraining:
- **A. original**;
- **B. leaf only** — background replaced by grey;
- **C. background only** — leaf replaced by grey. The mask is dilated by 8 px so the leaf
  outline is not left visible; shadows remain (a known leak of leaf shape);
- **D. background swap** — the leaf is pasted on the background of a random test image of
  *another* class (whose own leaf is greyed out). Measured: share of predictions that follow
  the leaf's class vs. the background's class.

**Hypotheses.**
1. Background-only accuracy (C) is well above chance (1/38 = 2.6%) for every model. [Noyan
   2022] reached ~49% with a model *trained* on backgrounds; ours never were, so expect 10–30%.
2. ConvNeXt-T and ViT-S, which look least at the leaf, are the most background-dependent: larger
   drop in B, higher accuracy in C, more predictions following the background in D.
3. ResNet-50 (88% leaf focus) keeps > 95% accuracy on leaf-only images (B).
4. Tomato bacterial spot is among the most background-dependent classes for ConvNeXt-T.

---

## 2026-10-20 — Grad-CAM: models mostly look at the leaf, but not only — and accuracy doesn't tell which

**Setup.** Whole test split; 10,096 images analysed (219 without a segmented twin, 394 where
the leaf covers >90% of the frame — mostly corn close-ups — excluded). Mean leaf area: 56.5%.
CAM of the predicted class.

| Model | Leaf focus | Pointing game | Random weights: focus / pointing | Accuracy |
|---|---:|---:|---:|---:|
| ResNet-50 | **88.2%** | **98.8%** | 58.5% / 44.8% | 99.44% |
| EfficientNet-B0 | 77.1% | 95.6% | 63.7% / 80.4% | 99.65% |
| ConvNeXt-T | 73.9% | 88.7% | 54.9% / 46.7% | 99.71% |
| ViT-S/16 | 68.8% | 67.9% | 35.3% / 32.2% | 99.73% |

**Hypotheses vs. results.**
1. *Most CAM mass on the leaf, well above the area baseline* → **confirmed** for all four
   (+12 to +32 points over leaf area), and above the random-weights baseline.
2. *>10% on the background* → **confirmed**: 12% (ResNet-50) to 31% (ViT-S).
3. *Errors have lower leaf focus* → **not supported.** Raw focus is lower for errors, but so is
   the leaf area of those images; focus minus area is similar for errors and correct
   predictions (e.g. ResNet-50 +0.320 vs +0.317). With 24–54 errors per model, no conclusion.
4. *ViT maps are noisier* → **confirmed** (pointing game 68% vs 89–99% for CNNs): its maps are
   patchy, made of many small 16×16-patch blobs (`reports/gradcam_examples.jpg`).

**The sanity check mattered.** A *randomly initialised* EfficientNet-B0 already puts its CAM
maximum on the leaf 80% of the time: part of what its map shows comes from the architecture and
the image (textured leaf on a flat background), not from training. Training adds only +13
points of leaf focus for EfficientNet-B0, vs +30 for ResNet-50. Without the random baseline we
would have over-credited EfficientNet's maps.

**The interesting dissociation.** The two most accurate models (ViT-S, ConvNeXt-T) are the
ones that look *least* at the leaf. On tomato *bacterial spot* (436 test images), ConvNeXt-T
focus is below the leaf area and its CAM maximum lands on the background half the time —
while classifying **all 436** correctly (recall 100%). Grad-CAM is correlational,
but this is exactly what a background shortcut [Noyan 2022] would look like.

**Pitfalls caught along the way.**
- Corn / healthy first came out with a pointing game of 0%: it had **1** analysable image
  (all others are full-frame close-ups). Per-class figures are only read for classes with ≥30 images.
- The ViT-S run hung on MPS (main thread blocked forever in Metal `waitUntilCompleted`);
  run on CPU instead with `--device cpu`.

**Takeaways.**
- High accuracy says nothing about *what* the model uses; the best models here are the most
  suspicious ones.
- Next step is causal, not visual: **replace the background** (masks make it easy) and measure
  how much accuracy drops per model and per class. That is the planned background-bias step.

---

## 2026-10-20 — Plan: does the model look at the leaf?

*Written before running the analysis.*

**Question.** A 99.7% accuracy does not tell us *why* the model is right. Does it look at the
leaf and its lesions, or partly at the background? [Noyan 2022] showed that PlantVillage
backgrounds carry class information, so a shortcut is plausible.

**Method.**
- Grad-CAM [Selvaraju et al. 2017] on the last feature map of each model (for ViT-S, the patch
  tokens of the last block, reshaped to the 14×14 grid).
- **Leaf mask for free**: PlantVillage ships a `segmented` version of each photo with the
  background blacked out. Mask = non-black pixels of the segmented twin (holes filled).
- Metrics on the test split, per image:
  - *leaf focus*: share of the Grad-CAM mass that falls on the leaf. Uniform map → leaf area
    fraction, so we compare against that baseline;
  - *pointing game* [Zhang et al. 2016]: is the CAM maximum on the leaf?
- **Sanity check** [Adebayo et al. 2018]: same metrics with randomly re-initialised weights.
  If a random model scores the same, the metric says nothing about what was learned.

**Hypotheses.**
1. All trained models put most of the CAM mass on the leaf, well above the area baseline.
2. A non-negligible part (>10%) still falls on the background, e.g. on shadows and leaf edges.
3. Misclassified images have lower leaf focus than correct ones.
4. ViT Grad-CAM maps are noisier (lower pointing-game score) than CNN maps.

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
*Correction (2026-10-20):* the idle-sleep explanation is not established. The next day, ViT-S
Grad-CAM hung on MPS *under* `caffeinate`, blocked in Metal `waitUntilCompleted`. The first
slowdown may have been the same intermittent MPS stall rather than the Mac sleeping.

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
