# Research log

Dated entries, newest first. Each entry: **question → what we did → result → takeaway**.
Negative or surprising results are kept on purpose.

---

## 2026-10-29 — Grounded generation: plan (before running it)

**Generator.** Qwen3-4B-Instruct-2507 (Apache 2.0), run locally (bf16, Apple GPU), greedy
decoding: free, no API key, reproducible. ~9 tokens/s on the development laptop.

**Prompt.** The top 5 chunks from bge-small (200 words), numbered [1]–[5] with their section
path; instructions: answer in at most 4 sentences using only the sources, put the source
number(s) after every sentence, and reply exactly "I don't know based on the available
sources." if they do not contain the answer. The prompt is fixed before the run; it may be
changed only to fix a format failure seen on dev.

**Conditions.**
- **retrieved**: the top 5 chunks, as the system would run.
- **oracle**: the same, but if no chunk contains a gold quote, the 5th is replaced by the gold
  chunk. The gap between the two is the cost of retrieval misses; what fails even with the
  gold chunk present is the generator's own failure (Joren et al. 2025).

**Metrics** (automatic, no LLM judge):
- **Abstention** on answerable vs unanswerable questions.
- **Citation format**: share of sentences carrying a valid citation.
- **Gold citation**: for answered answerable questions, does the answer cite a chunk that
  contains a gold quote?
- **Support (hallucination proxy)**: a sentence is supported if an NLI model
  (DeBERTa-v3-large, MNLI/FEVER/ANLI/LingNLI/WANLI) finds it entailed by one of the chunks it
  cites (probability > 0.5). Reported as the share of supported sentences. NLI is itself
  checked first: gold quotes against their own chunk must be entailed, against a random other
  chunk must not.

Not measured automatically: whether an answer is *correct and complete*. A sample of test
answers is read and described qualitatively.

**Hypotheses.**
- *G1*: ≥ 80% of cited sentences are supported by their citations (retrieved condition).
- *G2*: the model abstains on fewer than half of the unanswerable questions: given
  insufficient context, models hallucinate rather than abstain (Joren et al. 2025).
- *G3*: the oracle raises the gold-citation rate by about the retrieval miss rate, but not to
  100%: some failures are the generator's, even with the answer in front of it.

---

## 2026-10-28 — Retrieval experiments: big gains on dev, none on test for grower questions

**Dev** (17 grower + 18 extractive answerable questions; one grower question = 6 points),
recall@5 (`reports/rag/retrieval_experiments_dev.json`):

| Configuration (200-word chunks unless stated) | Extractive | Grower | All | All, 1,000-word budget |
|---|---:|---:|---:|---:|
| bge-small (baseline) | 1.00 | 0.71 | 0.86 | 0.91 |
| BM25 / BM25 + stemming | 1.00 | 0.65 / 0.59 | 0.83 / 0.80 | 0.86 / 0.80 |
| RRF(BM25, bge-small) | 1.00 | 0.76 | 0.89 | 0.94 |
| RRF(BM25 + stemming, bge-small) | 1.00 | 0.82 | 0.91 | 0.94 |
| bge-small → ms-marco-MiniLM reranker | 1.00 | 0.82 | 0.91 | 0.91 |
| bge-small → bge-reranker-base | 1.00 | 0.88 | 0.94 | 0.94 |
| **RRF(BM25 + stemming, bge-small) → bge-reranker-base** | 1.00 | **0.94** | **0.97** | 0.97 |
| bge-small, 100 / 400-word chunks | 0.94 / 1.00 | 0.47 / 0.71 | 0.71 / 0.86 | 0.89 / 0.89 |

**Hypotheses on dev.**
- *H5 (stemming helps BM25 on grower questions)* → **refuted**: +1 question at 100 words,
  −1 at 200 and 400. Snowball does not even merge *leaf* and *leaves*.
- *H2 (RRF beats both)* → **supported on dev**: +1 to +2 grower questions over bge-small.
- *H6 (chunk size matters little at an equal word budget)* → **supported**: 0.89–0.91 for
  bge-small at 1,000 words. At fixed k, 100-word chunks lose badly (0.71), which is only
  the budget effect. 200 words kept.
- *H7 (reranking +≥5 points on grower questions)* → **supported on dev**: +2 to +3 questions.

**Test**, run once, for the chosen pipeline and the baseline only
(`reports/rag/retrieval_experiments_test.json`):

| | Extractive | Grower | All (95% CI) | Recall@1 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| bge-small | 0.88 | **0.81** | 0.85 (0.75–0.94) | 0.58 | 0.69 |
| RRF(BM25 + stemming, bge-small) → bge-reranker-base | **1.00** | 0.77 | 0.88 (0.79–0.96) | **0.67** | **0.76** |

**The +4 grower questions found on dev turned into −1 on test.** The pipeline helps
extractive questions (all 26 found, 92% at rank 1) and the top of the ranking overall, but not
the grower questions it was chosen for. Overall: 1 vs 3 discordant questions, p = 0.63.

**Why.** Nineteen configurations were compared on 17 grower questions and the best one was
kept: its dev score was partly luck (the winner's curse). The decision rule asked for a gain of
two questions; the gain was four, and it still did not transfer. With sets this small,
**dev can rank big effects (BM25 vs dense on grower questions) but not 5–10 point ones.**

**Decision.** The retriever stays **bge-small alone** for now: simplest, no worse on grower
questions, and the reranker adds a second model to run per query. The hybrid + reranker
pipeline is kept in the code and will be re-evaluated when the evaluation set is larger: the
`human` questions, and more grower questions written for the dev split.

---

## 2026-10-28 — Retrieval experiments: plan (before running them)

All choices are made on **dev** (35 answerable questions); only the final pipeline and the
baseline (bge-small) are then run once on **test**. With so few dev questions, a difference of
one or two questions is noise: a change is kept only if it helps grower questions by at least
two questions and hurts nothing.

| # | Change | Hypothesis |
|---|---|---|
| E1 | BM25 + Snowball stemming | H5: +≥5 points recall@5 on grower questions, no loss on extractive |
| E2 | Hybrid: reciprocal rank fusion of BM25 and bge-small (k = 60) [Cormack et al. 2009] | H2: beats both on grower questions |
| E3 | Chunk size 100 / 200 / 400 words | H6: at an **equal budget of retrieved words**, size matters little (< 5 points) |
| E4 | Cross-encoder reranking of the top 20 (ms-marco MiniLM; bge-reranker-base) [Nogueira & Cho 2019] | H7: +≥5 points recall@5 on grower questions |

**Why an equal word budget for E3.** Recall@5 with 400-word chunks reads 2,000 words; with
100-word chunks, 500. Bigger chunks win at fixed k just by showing more text, which costs
prompt length later. E3 therefore compares recall at ~1,000 retrieved words
(k = 1000 / chunk size), as well as at fixed k.

---

## 2026-10-28 — Retrieval baselines: BM25 breaks on growers' words, dense models do not

Test split, recall@5 with 95% bootstrap intervals (`reports/rag/retrieval_baselines.json`):

| Retriever | All (n = 52) | Extractive (26) | Grower (26) | MRR@10 (all) |
|---|---:|---:|---:|---:|
| BM25 | 0.77 (0.65–0.88) | **0.96** | 0.58 (0.38–0.77) | 0.64 |
| all-MiniLM-L6-v2 | 0.71 | 0.81 | 0.62 | 0.52 |
| **bge-small-en-v1.5** | **0.85** (0.75–0.94) | 0.88 | **0.81** (0.65–0.96) | 0.69 |
| bge-m3 (dense) | 0.85 | 0.92 | 0.77 | 0.73 |

Dev split (17–18 questions per type): extractive recall@5 is 1.00 for every retriever (a
ceiling); grower 0.59–0.71, same ranking with smaller gaps.

**Hypotheses vs. results.**
1. *H1: BM25 within 5 points of bge-small overall* → **not supported, but not refuted
   either**: 8 points behind on test (3 on dev); the paired McNemar test finds no significant
   difference (4 vs 8 discordant questions, p = 0.39).
2. *H3: BM25 loses more than bge-small from extractive to grower questions* → **confirmed**.
   BM25: 0.96 → 0.58 (−38 points); bge-small: 0.88 → 0.81 (−7). The extra drop of BM25 is
   31 points (bootstrap 95% CI 8–54) on test, 21 (2–40) on dev + test pooled.

**What this means.** On questions that reuse the article's words (how synthetic evaluation sets
are usually built), BM25 is the best retriever here. On questions phrased like a grower's, it
is the worst but one. An evaluation set made only of synthetic questions would have picked
the wrong retriever, which is the risk the protocol was designed to expose.

**Other observations.**
- **Bigger is not better:** bge-m3 (568M parameters) does no better than bge-small (33M), and
  takes ~10x longer to index; all-MiniLM-L6-v2, the tutorial default, is the weakest model.
- **BM25 failures look fixable:** reading the 17 grower questions BM25 misses, many hinge on
  word forms (antibiotics / antibiotic, leaves / leaf, spraying / spray) that a stemmer would
  match, and a few on synonyms (bumps / pustules) that it would not. Next experiment, on dev.
- **The top-1 score separates answerable from unanswerable questions with AUROC ≈ 0.80** for
  all four retrievers: a usable but weak signal for abstention, to revisit in that step.
- Two grower questions are missed by every retriever (early blight overwintering, spore
  spread on tomato): candidates for checking whether the gold quote is the only answer.

---

## 2026-10-28 — Retrieval baselines: setup (written before running them)

**Retrievers**, all indexing the same text (section path + chunk, 200-word chunks):
- **BM25** (own implementation, k1 = 1.5, b = 0.75, English stopwords, no stemming).
- **all-MiniLM-L6-v2** (22M parameters, 2021): the default embedding model of many tutorials.
- **bge-small-en-v1.5** (33M): a small recent general-purpose model. **This is the "general
  dense model" of H1.**
- **bge-m3** (568M, dense mode only; Chen et al. 2024): the large multilingual option.

No model is tuned on this corpus; these are baselines.

**Metrics.** Recall@k: share of answerable questions with a relevant chunk (one containing a
gold quote) in the top k, for k = 1, 5, 10; MRR@10. Reported per question type on the **test**
split (dev numbers are kept for later choices), with 95% bootstrap intervals, and paired
McNemar tests on recall@5 hits between retrievers. Side measurement for the abstention step:
how well the top-1 score separates answerable from unanswerable questions (AUROC).

**Hypotheses** (from the kickoff, unchanged): H1 BM25 within 5 points of bge-small at
recall@5 overall; H3 BM25 loses more than bge-small from `extractive` to `grower` questions.
With 26 test questions per type, differences under ~15 points are unlikely to be significant:
the per-type results are indications, not proofs.

---

## 2026-10-28 — Evaluation set v1: 111 questions, the wording gap is real

Built following the protocol below, before any retriever exists
(`evaluation/questions.jsonl`, `reports/rag/evalset.json`).

| Type | Questions | Dev / test | Diseases | Question words found in evidence | Longest copied run |
|---|---:|---:|---:|---:|---:|
| `extractive` | 44 | 18 / 26 | 24 | 46% | 2.8 words |
| `grower` | 43 | 17 / 26 | 24 | **15%** | **1.2 words** |
| `unanswerable` | 24 | 10 / 14 | 18 | – | – |
| `human` | 0 | – | – | – | – |

- Every quote was found in the corpus and falls inside one chunk at 200 words (1.0 relevant
  chunk per question on average): labels are usable.
- The two answerable types differ as intended: grower questions share three times fewer
  content words with their evidence. H3 can now be tested.
- 24 of 26 diseases are covered by answerable questions; the 2 missing ones (esca, Isariopsis
  leaf spot) are the stubs, which appear only in unanswerable questions.
- 3 of the 46 passages sampled for grower questions were skipped (rust spore types, late blight
  mating types, spider mite inbreeding): nothing a grower would ask. A first, small measure of
  how much of an encyclopaedia is off-topic for this use.
- The unanswerable questions were checked by keyword search (doses, prices, pre-harvest
  intervals, approvals: zero hits), not proven unanswerable; a retriever may surface a
  passage that partly answers one, which will be inspected case by case.

**Limitation.** All 111 questions were written by an LLM in the development session. The
`human` subset, written by the author without looking at the corpus, is the check that the
grower questions are realistic and not just an LLM's idea of a grower; until it exists,
results on `grower` questions should be read with that caveat.

---

## 2026-10-28 — Evaluation set protocol (written before the questions)

**Why first.** Every retrieval choice (chunk size, BM25 vs dense, reranking) will be compared
on this set; writing it after seeing a retriever's output would bias it towards that retriever.

**Gold labels independent of chunking.** A question's gold evidence is one or more *quotes*:
exact sentences or clauses of an article. A retrieved chunk is relevant if it contains a quote.
Chunk size is an experimental variable, so gold labels must not be chunk ids.

**Question types.**
1. **`extractive`** (synthetic, the usual shortcut): written while reading one passage, as an
   LLM question generator does; reusing the passage's words is allowed.
2. **`grower`**: what a grower would type after the diagnosis: symptom words, plain language,
   crop named, disease named only by its common name. Written from a passage too, but the
   wording must not copy it. Overlap is *measured*, not trusted: share of the question's
   content words found in its evidence, and longest common word n-gram.
3. **`unanswerable`**: plausible questions about the 26 diseases that the corpus does not
   answer (stub articles, missing management sections, doses and products), plus a few
   off-topic ones. The right behaviour is to abstain.
4. **`human`**: written by the project author without looking at the corpus, then labelled.
   This is the only type not written by an LLM, and the reference for the others.

Types 1–3 are written by an LLM (Claude, in the development session) under the rules above;
they are labelled as such in the file. This is exactly the practice the literature uses and
rarely audits (Balaguer et al. 2024), so the `human` subset is there to measure the gap.

**Passages** are sampled with a fixed seed, two per disease and type when the corpus has
enough text, so that well-documented diseases (late blight: 34 chunks) do not dominate.

**Split.** 40% dev (to make choices), 60% test (to report), stratified by type, fixed by a
hash of the question id. Choices are made on dev only.

**Hypothesis (H3 of the kickoff, made measurable).** BM25 recall@5 drops more than dense
recall@5 from `extractive` to `grower` questions, because the lexical overlap with the
evidence is what BM25 relies on.

---

## 2026-10-28 — The corpus: what we may reuse decides what we can answer

**Question.** Which fact sheets, in which language, under which license?

**Sources considered** (licenses checked on 2026-10-28):

| Source | License / terms | Decision |
|---|---|---|
| PlantVillage website (disease pages) | CC BY-SA 3.0, but `robots.txt` blocks AI crawlers (GPTBot, ClaudeBot, …) | **Not used**: the license allows it, the site owner's stated wish does not |
| Ephytia (INRAE, French) | "tous droits réservés": INRAE's terms forbid reuse of such content | Not used |
| US university extension fact sheets | Copyright of each university, case-by-case permission | Not used |
| PlantwisePlus factsheets (CABI) | No license on the pages; related CABI material is CC BY-NC | Not used (unclear) |
| **English Wikipedia** | **CC BY-SA 4.0**, open API | **Used** |

Wikipedia is not the best agronomic source, but it is the only one that is clearly reusable,
redistributable with the repository and compatible with the CC BY-SA weights.

**Language.** English. French Wikipedia has an article for only 15 of the 28 English pages
and 43% of their text; Ephytia, the best French source, is not reusable.

**What we built.** 28 articles (one or two per disease, pathogen pages included), pinned to
their revision ids in `corpus/`, split into sections (references and links dropped), then into
chunks of at most 200 words that never cross a section boundary: **240 chunks, 28k words**.

**What the numbers show** (`reports/rag/corpus.json`):
- **Very uneven coverage.** Late blight: 4,452 words (most of it about the Irish famine);
  grape leaf blight (Isariopsis): **24 words**, a stub with no symptoms and no management;
  esca: 193 words.
- **4 of the 26 diseases have no management section** (esca, Isariopsis leaf spot, spider
  mites, tomato mosaic virus): for these the assistant cannot honestly answer "what can I do?".
- Pages are about the pathogen's biology more than about what a grower should do.

**Takeaways for the next steps.**
1. Unanswerable questions are not an edge case: the evaluation set must include questions the
   corpus cannot answer, and results must be reported per disease.
2. The assistant explains diseases from an encyclopaedia; it does not give treatment
   prescriptions (no product or dose advice), which is also the safer scope.
3. A new question for the vision ↔ RAG step: the classifier is equally confident on
   well-documented and barely documented diseases. The answer's reliability depends on both.

---

## 2026-10-28 — Phase 2 kickoff: what the literature leaves open

**Goal of Phase 2.** After a diagnosis, answer the user's follow-up questions ("how does it
spread?", "what can I do?") from agronomic fact sheets, citing them, and say "I don't know" when
the sheets do not cover the question.

**What recent agriculture RAG papers do** (references in `docs/references.md`):
- Systems that couple a vision model with RAG exist (coffee leaves with YOLOv8, Mondal et al.
  2025; AgriDoctor, Zhang et al. 2025), but their abstracts report **no evaluation of the
  retrieval step**: only end-to-end answers, or nothing.
- Evaluation sets are either **synthetic** (Q&A generated from PDFs and judged by GPT-4,
  Balaguer et al. 2024) or **expert-written but judged end-to-end** (Crop GraphRAG, Wu et al.
  2026). Nobody measures how far the synthetic questions are from real ones.
- In Crop GraphRAG, naive vector RAG beats BM25 + LLM, but on one corpus and with an LLM judge.
  BEIR (Thakur et al. 2021) found BM25 hard to beat out of domain: the question is open for a
  small, technical, vocabulary-heavy corpus like disease fact sheets.
- General RAG work points to three failure points we can measure separately: the chunk lost its
  context (contextual retrieval, late chunking), retrieval missed the passage, or the model had
  the passage and ignored it / answered without it (sufficient context, Joren et al. 2025).

**What LeafScan can add.** The classifier gives something most RAG systems lack: a **prior on
the topic** (the predicted disease) and a **measure of its uncertainty** (calibrated on field
photos in Phase 1). How to use that in retrieval is untested in the papers above.

**Design principles.**
1. Build the evaluation set before the system; keep a human-checked subset to audit any
   synthetic questions and any LLM judge (as ARES does with prediction-powered inference).
2. Evaluate retrieval on its own (recall@k against gold passages) before generation.
3. Report uncertainty: the evaluation set will be small, so bootstrap CIs, as in Phase 1.

**Hypotheses, written before any experiment.**
1. *BM25 is within 5 points of recall@5 of a general dense embedding model* on this corpus:
   disease and pathogen names are rare, exact terms that lexical matching handles well.
2. *Hybrid search (RRF) beats both* by a few points, because questions mix exact names
   ("early blight") with paraphrases ("brown rings on the leaves").
3. *Synthetic questions are easier than human-written ones*: they reuse the passage's words,
   so BM25 recall will be inflated on them more than dense recall.
4. *Filtering retrieval by the predicted disease helps when the classifier is confident and
   hurts when it is wrong*; retrieving for the top-2 classes when it hesitates recovers most of
   the loss.

**Next.** The corpus: which fact sheets, in which language, under which license.

---

## 2026-10-28 — Shipping it: quantisation and the browser change predictions a little

Hugging Face now requires a paid plan for Gradio Spaces, so the public demo runs the model in
the visitor's browser (ONNX Runtime Web) as a free static Space. Two transformations stand
between the evaluated model and what users get; both were measured on PlantDoc test (228 photos):

| Model | Size | Same top-1 as PyTorch | Accuracy | Answered / right (own threshold) |
|---|---:|---:|---:|---:|
| PyTorch (safetensors) | 106 MB | – | 72.4% | 65% / 87.2% |
| ONNX fp32 | 106 MB | 100% | 72.4% | 65% / 87.2% |
| ONNX int8 (Python) | 32 MB | 96.5% | 71.9% | 66% / 84.7% |
| ONNX int8 (browser) | 32 MB | 96.1% vs Python int8 | 71.1% | 66% / 86.8% |

- **Quantisation shifts predictions**, so the int8 model got its own temperature and threshold,
  refitted on the same field calibration split (the lesson of 2026-10-27: calibrate what you
  ship).
- Only linear layers are quantised: ONNX Runtime Web has no int8 convolution kernel (found when
  the first export failed to load in the browser).
- The exporter stored weights as Constant nodes, which the quantiser silently skipped: the first
  "int8" file was as large as fp32. Converting them to initializers fixed it.
- Browser vs Python differences come from image resizing (canvas vs PIL): 4% of top-1 change.
- Live check: the deployed page loads the model from the Hub and classifies PlantDoc test photos
  in 0.6–0.8 s on a laptop, correctly in the two cases tried.

---

## 2026-10-27 — The demo model: honest at 87% on two thirds of field photos

**Data.** PlantDoc train split into 1,975 training and 348 calibration images; 228 test images.

| | PlantDoc test | PlantVillage test |
|---|---:|---:|
| Accuracy | 72.4% (95% CI 66.2–78.1) | 96.6% (was 99.7%) |
| ECE before / after field temperature (T = 0.83) | 0.087 / 0.090 | – / 0.051 |
| With the field threshold (0.75): answered / right | **65% / 87%** | 94% / 98.7% |

Crop right: 86%; healthy vs diseased right: 95%. On the calibration set the threshold gives
64% coverage at 90.1% accuracy (the target); on test, 65% at 87.2%.

**Hypotheses vs. results.**
1. *Accuracy ≈ 73–75%* → **consistent**: 72.4%, just below the range, well within the ±6-point CI.
2. *Field temperature brings ECE below 0.05* → **refuted, for an interesting reason**: the
   field-fine-tuned model is already fairly calibrated on field photos (ECE 0.087 vs 0.24
   zero-shot), and a temperature fitted on 348 images does not improve it (0.090). Most of the
   miscalibration came from the domain gap, not from the confidence scale.
3. *The field threshold keeps ≥ 85% accuracy on ≥ 50% of photos* → **confirmed**: 87% on 65%.
   The 90% target set on calibration data lands at 87% on test, a small and honest shift.
4. *Forgetting stays above 90% on lab images* → **confirmed**: 96.6% (−3.1 points).

**Practical checks.** Predictor tried on real PlantDoc test photos: 2 of the 4 errors in a random
sample of 8 were flagged "Not sure". CPU latency with Grad-CAM: ~1.6 s per photo on the Mac CPU
(ConvNeXt's depthwise convolutions are slow there; see 2026-10-19). End-to-end call of the
running Gradio app through `gradio_client` returns the expected diagnosis and heatmap.

**Takeaway.** The final system is modest but honest: about 3 out of 4 field photos right; when it
answers, about 9 out of 10; when it doesn't know, it says so. That is a more useful claim than
"99.7%".

---

## 2026-10-27 — Plan: the model behind the demo

*Written before training it.*

**Goal.** One model for the demo, chosen and calibrated with what the previous steps taught:
field data matters most, and calibration must be fitted on field data.

**Setup.** ConvNeXt-T, started from the PlantVillage checkpoint, fine-tuned (400 steps, same
recipe as the field fine-tuning study) on 85% of PlantDoc train; the other 15% (stratified) is a
**field calibration set** used only to fit the temperature and choose the abstention threshold
(lowest confidence whose answers are >= 90% right on the calibration set). Reported on PlantDoc
test (never used before for any choice) and on PlantVillage test.

**Hypotheses.**
1. PlantDoc test accuracy about 73-75% (slightly below the 74.4% of the full-train runs: 15% less data).
2. Field-fitted temperature brings PlantDoc test ECE below 0.05 (vs 0.24 zero-shot, 0.49 with the
   lab temperature).
3. The field threshold keeps >= 85% accuracy on the PlantDoc photos it answers, while answering
   at least half of them.
4. **Forgetting**: PlantVillage test accuracy drops from 99.7% but stays above 90%: lab images
   are easy and the field fine-tuning is short.

---

## 2026-10-27 — "I don't know" works in the lab and breaks in the field

| Model | Lab ECE (→ after T) | Field ECE (→ after lab T) | Field acc. at 100 / 50 / 10% coverage | Lab threshold on field: answered / accuracy | AUROC lab leaves vs flowers / textures | AUROC field leaves vs flowers / textures |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-T | 0.098 → 0.001 | 0.240 → **0.487** | 31.2 / 43.6 / 61.2% | 19.7% / 56.1% | 0.999 / 0.999 | 0.83 / 0.90 |
| ViT-S/16 | 0.099 → 0.001 | 0.246 → 0.488 | 29.7 / 42.6 / 54.9% | 15.8% / 53.7% | 0.997 / 0.997 | 0.80 / 0.80 |
| ResNet-50 | 0.094 → 0.002 | 0.227 → 0.471 | 27.0 / 36.9 / 52.9% | 20.4% / 48.7% | 0.983 / 0.981 | 0.67 / 0.68 |
| EfficientNet-B0 | 0.104 → 0.002 | 0.229 → 0.515 | 23.4 / 31.2 / 52.2% | 13.9% / 50.4% | 0.992 / 0.986 | 0.62 / 0.71 |

ECE: 15 bins. AUROC with maximum softmax probability (energy in `reliability.json`).
Temperature fitted on PlantVillage validation: T = 0.53–0.56. Figure: `reports/reliability.png`.

**Hypotheses vs. results.**
1. *Underconfident in the lab, fixed by temperature scaling* → **confirmed**: mean confidence
   0.89–0.90 at 99.4–99.7% accuracy (ECE ≈ 0.10); after T ≈ 0.55, ECE 0.001–0.002.
2. *Overconfident in the field, and the lab temperature makes it worse* → **confirmed**:
   confidence 0.46–0.55 at 23–31% accuracy (ECE 0.23–0.25). The lab temperature *sharpens*
   predictions, which **doubles** field ECE (0.47–0.52). Calibration fitted in one domain is
   harmful in another.
3. *Answering on the 50% most confident field photos gives ≥ 45%* → **refuted**: 31–44%
   (+8 to +12 points). Abstention helps, but slowly: even the 10% most confident field
   answers are only 52–61% right.
4. *Lab thresholds don't transfer* → **confirmed**: the threshold that gives 99.9% accuracy on
   lab validation answers 98–99% of lab test photos at 99.8–99.96%, but only 14–20% of field photos,
   and those answers are right **49–56%** of the time.
5. *Textures easy (> 0.9), flowers hard (< 0.8)* → **half refuted.** Against *lab* leaves, both
   are trivially rejected (AUROC 0.98–0.999), flowers included. Against *field* leaves, which is
   what a real app sees, rejection collapses: 0.62–0.83 for flowers, 0.68–0.90 for textures.
   And field leaves themselves look out-of-distribution: lab vs PlantDoc AUROC is 0.94–0.96.
   The energy score is not consistently better than softmax confidence here.

**Takeaways.**
- Every safety mechanism measured in the lab (calibration, confidence threshold, OOD rejection)
  looked excellent and **did not survive the move to field photos**. Lab validation cannot
  certify field reliability; only field data can.
- For the demo: no lab-tuned threshold. Either fine-tune and calibrate on field data, or display
  confidence honestly ("low confidence — consult an expert") with a threshold chosen on PlantDoc.
- ConvNeXt-T is again the best compromise (best field accuracy, best field OOD rejection).

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
