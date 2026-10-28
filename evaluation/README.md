# Retrieval evaluation set

`questions.jsonl`: one question per line. Protocol and rationale: research log, 2026-10-28.

| Field | Meaning |
|---|---|
| `id` | `ext-*`, `grw-*`, `una-*`, `hum-*` |
| `type` | `extractive`, `grower`, `unanswerable` or `human` |
| `question` | the question, in English |
| `classes` | PlantVillage class(es) the question is about (what the classifier would predict) |
| `evidence` | exact quotes from `corpus/wikipedia.jsonl` that answer it; empty if unanswerable |
| `source_chunk` | passage it was written from (chunks of 200 words), for provenance only |
| `origin` | who wrote it |

A retrieved chunk is **relevant** if it contains one of the quotes (case and whitespace
ignored), so the labels hold for any chunk size. The dev/test split (40/60, stratified by type)
is computed from the ids by `leafscan.rag.evalset.assign_splits`; experiments choose on dev and
report on test.

Validate after any edit:

```bash
python -m leafscan.rag.evalset check
```

## Adding `human` questions

Write the questions **before reading the corpus**, as you would ask them about a sick plant,
one per line with `"type": "human"`, an id `hum-NN` and the disease in `classes`. Evidence
quotes are added afterwards by searching the corpus; a question the corpus cannot answer keeps
`"evidence": []` and becomes `"type": "unanswerable"` with `"origin"` saying it was written by
a human.

Text quoted in `evidence` comes from Wikipedia (CC BY-SA 4.0, see `corpus/README.md`).
