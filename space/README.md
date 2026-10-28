---
title: LeafScan
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Plant disease from a leaf photo, runs in your browser
models:
  - Marconcini/leafscan-convnext-tiny
---

# LeafScan

Upload a leaf photo: top-3 diagnoses with calibrated probabilities, and a **"Not sure"** answer
when confidence is below a threshold fitted on field photos. The model (ConvNeXt-T, int8 ONNX,
32 MB) runs **entirely in your browser** with ONNX Runtime Web: nothing is uploaded.

About 72% right on held-out field photos; it answers about two thirds of them, ~85% of those
right. A research demo, not a diagnostic tool.

Code, experiments and limitations: [github.com/marconcini/leafscan](https://github.com/marconcini/leafscan) ·
Model: [Marconcini/leafscan-convnext-tiny](https://huggingface.co/Marconcini/leafscan-convnext-tiny)

The Python/Gradio version of the demo (with Grad-CAM) runs locally: `python -m leafscan.app`.
