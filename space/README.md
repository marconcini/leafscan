---
title: LeafScan
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Plant disease from a leaf photo, honest about uncertainty
models:
  - Marconcini/leafscan-convnext-tiny
---

# LeafScan

Upload a leaf photo: top-3 diagnoses with calibrated probabilities, a Grad-CAM map, and a
**"Not sure"** answer when confidence is below a threshold fitted on field photos.

72% right on held-out field photos; it answers about two thirds of them, ~87% of those right.
A research demo, not a diagnostic tool.

Code, experiments and limitations: [github.com/marconcini/leafscan](https://github.com/marconcini/leafscan) ·
Model: [Marconcini/leafscan-convnext-tiny](https://huggingface.co/Marconcini/leafscan-convnext-tiny)
