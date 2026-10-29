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

## Beyond this page: a cited assistant (local only)

The full project adds a question-answering assistant after the diagnosis. It retrieves passages
from 28 Wikipedia articles, answers with a local LLM (Qwen3-4B) citing each source, says
"I don't know" when the sources do not answer, and flags with ⚠️ the sentences an NLI model
finds unsupported. Its models are ~10 GB, so it runs on your machine, not in this page:

```bash
pip install -e ".[app,rag]"
LEAFSCAN_ASSISTANT=1 python -m leafscan.app     # photo → diagnosis → cited answers, Grad-CAM
```

What the evaluation showed: the assistant abstains on 86% of unanswerable test questions and
78% of its cited sentences are supported by their source; letting the diagnosis filter the
search would lose 16 points of recall, because a wrong diagnosis removes the answer. Details
in the [README](https://github.com/marconcini/leafscan#phase-2--rag-assistant) and the
[research log](https://github.com/marconcini/leafscan/blob/main/docs/research-log.md).
