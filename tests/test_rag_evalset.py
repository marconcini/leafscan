import pytest

from leafscan.rag.corpus import chunk_pages
from leafscan.rag.evalset import (
    assign_splits,
    check,
    longest_common_ngram,
    overlap,
    relevant_chunks,
    sample_passages,
)

PAGES = [{"title": "Leaf rot", "url": "u", "classes": ["Tomato___Leaf_Mold"], "text": (
    "Leaf rot is a disease of tomato.\n\n\n"
    "== Symptoms ==\nOlive-green spots appear on the underside of older leaves.\n\n\n"
    "== Management ==\nGreenhouses should be ventilated to keep humidity below 85%.\n")}]


def q(qid, kind, question, evidence=()):
    return {"id": qid, "type": kind, "question": question, "evidence": list(evidence),
            "classes": ["Tomato___Leaf_Mold"]}


def test_quotes_find_their_chunk_whatever_the_chunk_size():
    question = q("g1", "grower", "Why is the back of my tomato leaves going green?",
                 ["olive-green spots appear on the  underside"])
    for max_words in (5, 200):
        chunks = chunk_pages(PAGES, max_words)
        hits = relevant_chunks(question, chunks)
        assert len(hits) == 1 and "underside" in next(c["text"] for c in chunks
                                                      if c["id"] == hits[0])


def test_overlap_measures_copied_wording():
    ev = ["Greenhouses should be ventilated to keep humidity below 85%."]
    copied = overlap("Should greenhouses be ventilated to keep humidity below 85%?", ev)
    paraphrased = overlap("My plants are under plastic and it is muggy, what helps?", ev)
    assert copied["content_word_overlap"] == 1.0 and copied["longest_common_ngram"] >= 5
    assert paraphrased["content_word_overlap"] == 0.0
    assert longest_common_ngram("a b c d", "x b c y") == 2


def test_check_accepts_a_valid_set_and_reports_broken_labels():
    good = [q("e1", "extractive", "What appears on the underside of older leaves?",
              ["Olive-green spots appear on the underside of older leaves."]),
            q("u1", "unanswerable", "Which fungicide dose should I spray?")]
    res = check(good, PAGES)
    assert res["per_type"]["extractive"]["relevant_chunks_mean"] == 1
    with pytest.raises(ValueError, match="quote not found"):
        check([q("e2", "extractive", "?", ["purple spots"])], PAGES)
    with pytest.raises(ValueError, match="iff"):
        check([q("u2", "unanswerable", "?", ["Leaf rot is a disease of tomato."])], PAGES)
    with pytest.raises(ValueError, match="duplicate"):
        check(good + good[:1], PAGES)


def test_splits_are_deterministic_stratified_and_stable():
    qs = [{"id": f"e{i}", "type": "extractive"} for i in range(50)] + \
         [{"id": f"u{i}", "type": "unanswerable"} for i in range(10)]
    splits = assign_splits(qs)
    assert splits == assign_splits(list(reversed(qs)))
    assert sum(splits[f"e{i}"] == "dev" for i in range(50)) == 20
    assert sum(splits[f"u{i}"] == "dev" for i in range(10)) == 4
    more = assign_splits(qs + [{"id": "u10", "type": "unanswerable"}])
    assert all(more[f"e{i}"] == splits[f"e{i}"] for i in range(50))


def test_sampled_passages_are_unique_and_grower_ones_come_from_relevant_sections():
    chunks = chunk_pages(PAGES, max_words=200)
    out = sample_passages(chunks, per_class=1, min_words=1)
    ids = [p["id"] for ps in out.values() for p in ps]
    assert len(ids) == len(set(ids)) == 2
    assert all(p["section"] != "Leaf rot" for p in out["grower"])
