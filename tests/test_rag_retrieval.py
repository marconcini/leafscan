import numpy as np

from leafscan.rag.retrieval import BM25, Dense
from leafscan.rag.retrieval_eval import mcnemar_hits, metrics, run

CHUNKS = [
    {"id": "a#0", "section": "Early blight > Symptoms", "text": "Concentric rings on old leaves."},
    {"id": "b#0", "section": "Late blight > Management", "text": "Destroy infected tubers."},
    {"id": "c#0", "section": "Leaf mold > Management", "text": "Keep humidity low. Ventilate."},
]


def test_bm25_prefers_rare_matching_terms_and_uses_section_names():
    bm25 = BM25(CHUNKS)
    assert bm25.search("rings on leaves", k=1)[0][0] == "a#0"
    assert bm25.search("late blight", k=1)[0][0] == "b#0"     # only in the section path
    assert all(v > 0 for v in bm25.idf.values())
    assert bm25.scores("zzz unknown").sum() == 0


def toy_encoder(texts):
    """Bag of words over a tiny vocabulary: enough to test the plumbing."""
    vocab = ["rings", "tubers", "humidity", "blight", "mold"]
    return np.array([[t.lower().count(w) + 1e-3 for w in vocab] for t in texts])


def test_dense_ranks_by_cosine_similarity():
    dense = Dense(CHUNKS, "toy", encode=toy_encoder)
    assert dense.search("humidity", k=1)[0][0] == "c#0"
    assert np.allclose(np.linalg.norm(dense.matrix, axis=1), 1)


def test_run_and_metrics():
    questions = [
        {"id": "q1", "type": "grower", "split": "test", "question": "rings on my leaves",
         "evidence": ["Concentric rings"]},
        {"id": "q2", "type": "grower", "split": "test", "question": "humidity",
         "evidence": ["Destroy infected tubers."]},
        {"id": "q3", "type": "unanswerable", "split": "test", "question": "price?",
         "evidence": []},
    ]
    rows = run(BM25(CHUNKS), questions, CHUNKS, k=2)
    assert [r["rank"] for r in rows] == [1, None, None]
    m = metrics(rows)
    assert m["n"] == 2 and m["recall@1"] == 0.5 and m["mrr@10"] == 0.5
    assert 0 <= m["answerable_auroc"] <= 1


def test_mcnemar_counts_discordant_questions():
    a = [{"id": "q1", "rank": 1}, {"id": "q2", "rank": None}, {"id": "q3", "rank": 2}]
    b = [{"id": "q1", "rank": None}, {"id": "q2", "rank": 3}, {"id": "q3", "rank": 1}]
    res = mcnemar_hits(a, b)
    assert (res["only_first"], res["only_second"]) == (1, 1) and res["p"] == 1.0
