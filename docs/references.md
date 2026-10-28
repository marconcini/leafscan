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

| Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997 | Map of the field: naive / advanced / modular RAG, evaluation. |
| Saad-Falcon et al. (2024). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* NAACL. arXiv:2311.09476 | LLM judges corrected by a small human-labelled set: how to trust automatic RAG metrics. |
| Chen et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216 | Multilingual dense + sparse embeddings in one model: candidate if the corpus mixes French and English. |
| Günther et al. (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models.* arXiv:2409.04701 | Chunks lose their document's context; embed the document first, split afterwards. |
| Anthropic (2024). *Introducing Contextual Retrieval.* anthropic.com/news/contextual-retrieval | Same problem, solved by prepending an LLM-written context to each chunk; claims −49% failed retrievals (−67% with reranking), on the authors' own benchmarks. |
| Joren et al. (2025). *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.* ICLR. arXiv:2411.06037 | Separates "retrieval missed it" from "the model ignored it"; models hallucinate rather than abstain when context is insufficient. |

### Agriculture

| Reference | Why it matters here |
|---|---|
| Balaguer et al. (2024). *RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture.* arXiv:2401.08406 | Q&A pairs generated from agronomic PDFs and judged by GPT-4; RAG and fine-tuning gains add up. Their synthetic evaluation set is what we want to check against human labels. |
| Mondal et al. (2025). *Vision Meets Language: A RAG-Augmented YOLOv8 Framework for Coffee Disease Diagnosis and Farmer Assistance.* arXiv:2505.21544 | Closest to LeafScan: detector output feeds a RAG pipeline. The abstract reports no evaluation of the RAG part. |
| Zhang et al. (2025). *AgriDoctor: A Multimodal Intelligent Assistant for Agriculture.* arXiv:2509.17044 | Router + classifier + knowledge retriever over 831 expert-written entries; retrieval itself not described in the abstract. |
| Wu et al. (2026). *Crop GraphRAG: pest and disease knowledge base Q&A system for sustainable crop protection.* Frontiers in Plant Science | 200+ expert-written questions; naive vector RAG vs BM25 vs knowledge-graph RAG. A human-built evaluation set, but answers are judged end-to-end only. |
