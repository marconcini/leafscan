---
license: cc-by-sa-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - plant-disease
  - agriculture
  - convnext
  - timm
datasets:
  - mohanty/PlantVillage
metrics:
  - accuracy
---

# LeafScan — ConvNeXt-T for plant disease classification (field-calibrated)

Classifies a leaf photo into **38 crop / disease classes** (14 crops) and says **"Not sure"**
when its confidence is too low. Research / portfolio project:
[github.com/marconcini/leafscan](https://github.com/marconcini/leafscan) — every number below
comes from experiments documented there, with hypotheses written before results.

- **Architecture:** ConvNeXt-T (`timm`: `convnext_tiny.fb_in22k_ft_in1k`), 28M parameters, 224×224 input.
- **Training:** ImageNet weights → PlantVillage (lab photos, 39k images) → 85% of PlantDoc train
  (1,975 web/field photos). The other 15% of PlantDoc train (348 photos) is used only to fit a
  temperature (T = 0.83) and an abstention threshold (0.75 on calibrated confidence).

## Results

| | PlantDoc test (field, 228 photos) | PlantVillage test (lab, 10,709 photos) |
|---|---:|---:|
| Accuracy | **72.4%** (95% CI 66.2–78.1) | 96.6% |
| Right crop | 86.4% | – |
| Healthy vs diseased | 94.7% | – |
| Answers above the threshold | 65% of photos, **87%** of them right | 94% of photos, 98.7% right |

For context: the same architecture trained on PlantVillage only reaches 99.7% on lab photos but
**31%** on field photos. Lab accuracy says little about real-world use.

## How to use

```bash
pip install "leafscan @ git+https://github.com/marconcini/leafscan.git"
```

```python
from PIL import Image
from leafscan.predictor import Predictor

predictor = Predictor("Marconcini/leafscan-convnext-tiny")   # downloads config.json + model.safetensors
pred = predictor(Image.open("leaf.jpg"))
print(pred.verdict)   # diagnosis, or "Not sure" below the field-calibrated threshold
print(pred.top)       # top-3 (class, calibrated probability)
```

Files:
- `model.safetensors` — PyTorch weights; `config.json` — class names, input size, temperature,
  threshold (and those of the int8 model under `onnx_int8`).
- `model.onnx` — fp32 ONNX export, identical top-1 to PyTorch on PlantDoc test.
- `model_int8.onnx` — 32 MB, linear layers quantised to int8 (convolutions kept fp32, as ONNX
  Runtime Web has no int8 convolution). Used by the in-browser demo. Same predictions as fp32
  on 96.5% of PlantDoc test photos; 71.9% accuracy. Because quantisation changes some
  predictions, it has **its own temperature and threshold**, refitted on the same field
  calibration set: on PlantDoc test it answers 66% of photos, 84.7% of them right.

In the browser (ONNX Runtime Web, image resizing done by the canvas instead of PIL), the demo
agrees with the Python int8 predictions on 96.1% of PlantDoc test photos.

## Intended use and limitations

- **Intended use:** demonstration and research on robustness of plant-disease classifiers.
  **Not** a diagnostic tool and not a substitute for an agronomist.
- Knows only the 38 PlantVillage classes; any other plant or disease is forced into one of them
  (or answered "Not sure").
- Test set is small (228 photos): differences under ~6 points are within noise.
- Calibration and threshold were fitted on PlantDoc photos; they may not hold for photos from
  other sources (the project shows lab-fitted thresholds failing on field photos).
- Rejection of non-leaf images is weak next to real field photos (AUROC 0.6–0.8 in the
  project's tests): a photo of something else may still receive a diagnosis.
- PlantDoc is web-scraped and contains label noise (duplicates filed under different classes
  were removed).

## Training data and licenses

- **PlantVillage** (Hughes & Salathé, 2015), CC BY-SA 3.0 — via `mohanty/PlantVillage`.
- **PlantDoc** (Singh et al., 2020), CC BY 4.0 — [github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset).
- Weights released under **CC BY-SA 4.0** (share-alike, following PlantVillage's license).
  Code: MIT.

## References

- Hughes & Salathé (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060
- Singh et al. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS-COMAD
- Liu et al. (2022). *A ConvNet for the 2020s.* CVPR
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML
