# Roadmap

Each step is small, testable and pushed on its own. Research questions are tracked in
[`docs/research-log.md`](docs/research-log.md), papers in [`docs/references.md`](docs/references.md).

## Phase 1 — Vision

- [x] **Data pipeline** — PlantVillage download, leak-free validation split by physical leaf
- [x] **Training & evaluation pipeline** — fine-tuning loop, metrics, confusion matrix, CI
- [x] **Baseline** — EfficientNet-B0 on the full dataset (99.65% test accuracy)
- [x] **Model comparison** — ResNet-50 · ConvNeXt-T · ViT-S: accuracy, macro-F1, latency, size
  - *Q: does a Vision Transformer bring anything on such a "clean" lab dataset?*
- [x] **Error analysis & Grad-CAM** — leaf focus vs leaf masks, random-weights sanity check
  - *Q: does the model actually look at the lesions?*
- [x] **Background bias** — causal test by editing backgrounds with the leaf masks: models do not
  use the background (background-only ≈ chance once silhouette and default-class artefacts are
  removed); EfficientNet-B0 is the most disturbed by background changes, ViT-S the least
- [ ] **Training on the `segmented` variant** — does it make models more robust to background changes?
- [x] **Lab → field domain shift** — zero-shot on PlantDoc [Singh et al. 2020]: 99.7% → 23–31%;
  ConvNeXt-T ≈ ViT-S > ResNet-50 > EfficientNet-B0 (as the background test predicted for the last)
- [ ] **Closing the gap** — fine-tune on PlantDoc's train split, evaluate on its test split
  - *Q: how many in-domain images are needed? Does lab pre-training still help?*
- [ ] **Ensembling** — models make different errors (Jaccard 0.17–0.41): does averaging them help?
- [ ] **Data cleaning** — find mislabelled / empty images (2 found by hand among errors shared by all models)
- [ ] **Calibration & out-of-distribution rejection** — temperature scaling [Guo et al. 2017]
  - *Q: what does the model say about a photo that is not a leaf?*
  - *Field confidence is low on errors (2–4% of errors > 0.9): can abstention recover precision?*
- [ ] **Demo** — Gradio app, weights on the Hugging Face Hub, Space
  - *Q: does CPU latency of depthwise models change with ONNX Runtime or on x86 (oneDNN)?*

## Phase 2 — RAG assistant (research-oriented)

Principle: **build the evaluation before the system**, then improve it through experiments,
each with a hypothesis, a metric and a conclusion.

- [ ] **Literature review** and research log setup [Lewis et al. 2020; Karpukhin et al. 2020]
- [ ] **Corpus** — agronomic fact sheets per disease: sources, licenses, cleaning, chunking
  - *Q: French or English sources? Which ones may we legally reuse?*
- [ ] **Evaluation set** — questions with gold passages, partly LLM-generated then human-checked
  - *Q: can we trust synthetic questions?*
- [ ] **Retrieval baselines** — BM25 vs dense embeddings [Robertson & Zaragoza 2009; Thakur et al. 2021]
- [ ] **Retrieval experiments** — chunk size, hybrid search with RRF [Cormack et al. 2009],
  cross-encoder reranking [Nogueira & Cho 2019], HyDE [Gao et al. 2022], contextual retrieval
- [ ] **Grounded generation with citations** and hallucination measurement [Es et al. 2023]
  - *Q: does an LLM judge agree with human judgement? [Zheng et al. 2023]*
- [ ] **Knowing when to say "I don't know"** [Asai et al. 2023; Yan et al. 2024]
- [ ] **Vision ↔ RAG integration**
  - *Q: when the classifier hesitates between two diseases, should retrieval cover both?*
- [ ] **Final app and write-up**
