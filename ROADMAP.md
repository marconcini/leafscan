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
- [x] **Closing the gap** — fine-tuning on PlantDoc: 38.6% → 74–76% on its test split; PlantVillage
  pre-training helps only with very little field data (233 images: +8 points)
- [ ] **More field data** — accuracy still rises ~5 points per doubling: other field datasets?
- [ ] **Ensembling** — models make different errors (Jaccard 0.17–0.41): does averaging them help?
- [ ] **Data cleaning** — find mislabelled / empty images (2 found by hand among errors shared by all models)
- [x] **Calibration, abstention & OOD rejection** — all work in the lab (ECE 0.001 after temperature
  scaling, OOD AUROC ≥ 0.98) and break on field photos (lab temperature doubles field ECE; lab
  threshold: 99.9% → ~53%; flowers vs field leaves AUROC 0.62–0.83)
- [x] **Demo model** — PlantVillage → PlantDoc fine-tuning with a field calibration split:
  72.4% on PlantDoc test; with the field threshold, 65% answered at 87% accuracy
- [x] **Demo app** — Gradio: calibrated top-3, "Not sure" below the field threshold, Grad-CAM
- [x] **Publish** — weights (safetensors + ONNX) on the Hugging Face Hub; in-browser demo as a
  free static Space (Gradio Spaces now need a paid plan)
  - *Q: does CPU latency of depthwise models change with ONNX Runtime or on x86 (oneDNN)?*

## Phase 2 — RAG assistant (research-oriented)

Principle: **build the evaluation before the system**, then improve it through experiments,
each with a hypothesis, a metric and a conclusion.

- [x] **Literature review** — recent agriculture RAG systems rarely evaluate retrieval on its own
  or check synthetic questions against real ones; four hypotheses written before experiments
- [x] **Corpus** — 28 English Wikipedia articles (CC BY-SA), 240 section-aware chunks; the only
  clearly reusable source. Coverage is very uneven: 4 diseases have no management section
  - *Q: French or English sources? Which ones may we legally reuse?* → English; see research log
- [x] **Evaluation set v1** — 111 questions (extractive, grower-style, unanswerable) with quote-based
  gold labels that hold for any chunking; grower questions copy 3x less wording from the evidence
  - [ ] `human` subset written by the author without reading the corpus
  - *Q: can we trust synthetic questions?*
- [x] **Retrieval baselines** — BM25 vs dense embeddings [Robertson & Zaragoza 2009; Thakur et al. 2021]:
  BM25 is best on extractive questions (0.96 recall@5) and drops to 0.58 on grower questions;
  bge-small-en-v1.5 holds 0.81. Synthetic-only evaluation would have picked the wrong retriever
- [x] **Retrieval experiments** — chunk size, hybrid search with RRF [Cormack et al. 2009],
  cross-encoder reranking [Nogueira & Cho 2019]: +4 grower questions on dev, −1 on test
  (winner's curse on 17 questions); bge-small kept. Next: a bigger dev set before more tuning
- [ ] HyDE [Gao et al. 2022], contextual retrieval — after the evaluation set grows
- [x] **Grounded generation with citations** and hallucination measurement [Es et al. 2023] —
  local Qwen3-4B; abstains on 88% of unanswerable questions; 78% of cited sentences entailed
  (NLI), ~12% estimated real embellishments, mostly in grower answers
  - *Q: does an LLM judge agree with human judgement? [Zheng et al. 2023]*
- [x] **Knowing when to say "I don't know"** [Asai et al. 2023; Yan et al. 2024] — the model's own
  abstention does most of the work; a retrieval gate and an NLI filter each catch one more
  question for ~10% of answers lost. Unsupported sentences are flagged, not removed
  - [ ] question-level "sufficient context" check [Joren et al. 2025]
- [x] **Vision ↔ RAG integration** — simulated on 500 (question, field photo) pairs: filtering
  retrieval by the diagnosis costs 16 points of recall@5 (a wrong diagnosis removes the answer);
  top-2 when unsure recovers less than half; even prepending a *right* disease name hurts.
  The app retrieves with the question as written and uses the diagnosis to suggest questions
  - *Q: when the classifier hesitates between two diseases, should retrieval cover both?* →
    no: suggest questions about both instead
- [x] **Final app and write-up** — README with a two-minute summary; local Gradio app with the
  assistant (`LEAFSCAN_ASSISTANT=1`)
