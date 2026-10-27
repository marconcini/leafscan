# References

What each paper brings to this project. Grouped by phase; extended as the project progresses.

## Phase 1 — Vision

| Reference | Why it matters here |
|---|---|
| Hughes & Salathé (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060 | The PlantVillage dataset. |
| Mohanty, Hughes & Salathé (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science | Reference result (~99% accuracy) and the warning that performance drops on images from other conditions. |
| Singh et al. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS-COMAD | Field photos, to measure the lab → field gap. |
| Noyan (2022). *Uncovering bias in the PlantVillage dataset.* arXiv:2206.04374 | Shows that background pixels alone are predictive → our models may learn shortcuts. |
| He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR | ResNet-50. |
| Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML | EfficientNet-B0 baseline. |
| Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR | ViT. |
| Liu et al. (2022). *A ConvNet for the 2020s.* CVPR | ConvNeXt. |
| Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV | Explaining predictions. |
| Zhang et al. (2016). *Top-down Neural Attention by Excitation Backprop.* ECCV | The "pointing game" to evaluate saliency maps against object locations. |
| Adebayo et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS | Saliency maps can look plausible even for random weights → always compare to a randomised model. |
| Dietterich (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation | McNemar's test to compare two classifiers on one test set. |
| Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML | Temperature scaling; are confidence scores trustworthy? |
| Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural Networks.* NeurIPS | Abstaining on uncertain inputs; risk–coverage curves. |
| Hendrycks & Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR | Maximum softmax probability as an OOD score. |
| Liu et al. (2020). *Energy-based Out-of-distribution Detection.* NeurIPS | Energy score, usually better than softmax confidence for OOD. |
| Nilsback & Zisserman (2008). *Automated Flower Classification over a Large Number of Classes.* ICVGIP | Oxford Flowers-102, near-OOD images. |
| Cimpoi et al. (2014). *Describing Textures in the Wild.* CVPR | DTD, far-OOD images. |

## Phase 2 — RAG

| Reference | Why it matters here |
|---|---|
| Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS | The RAG paradigm. |
| Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP | Dense retrieval baseline. |
| Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* | Lexical baseline. |
| Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks | BM25 remains a strong baseline out of domain. |
| Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR | Hybrid search. |
| Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085 | Cross-encoder reranking. |
| Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). arXiv:2212.10496 | Query expansion with hypothetical documents. |
| Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024 | Position of retrieved passages in the prompt. |
| Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024 (demo) | Faithfulness / relevance metrics. |
| Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks | Biases of LLM judges. |
| Asai et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024 | Deciding when to retrieve / abstain. |
| Yan et al. (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884 | Detecting and handling poor retrieval. |

> Recent work (2025–2026) will be added during the Phase 2 literature review.
