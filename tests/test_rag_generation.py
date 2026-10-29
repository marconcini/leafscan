import numpy as np

from leafscan.rag.attribution import check_answer, sanity_check, summarize
from leafscan.rag.generate import ABSTAIN, build_messages, oracle_context, parse

CHUNKS = [
    {"id": "a#0", "page": "A", "section": "A > Symptoms", "text": "Spots have concentric rings."},
    {"id": "b#0", "page": "B", "section": "B > Control", "text": "Remove infected debris."},
    {"id": "c#0", "page": "C", "section": "C", "text": "Whiteflies spread the virus."},
]
QUESTION = {"id": "g1", "type": "grower", "question": "What do the spots look like?",
            "evidence": ["concentric rings"]}


def word_overlap_nli(pairs):
    """Toy NLI: 'entailed' when every word of the hypothesis is in the premise."""
    def bag(text):
        return set(text.lower().replace(".", " ").split())

    return np.array([float(bag(h) <= bag(p)) for p, h in pairs])


def test_parse_citations_and_abstention():
    p = parse("Spots have concentric rings [1]. Remove debris [2][7]. No citation here.", 3)
    assert [s["cited"] for s in p["sentences"]] == [[1], [2, 7], []]
    assert [s["valid"] for s in p["sentences"]] == [True, False, False]
    assert p["sentences"][0]["text"] == "Spots have concentric rings ."
    assert parse(ABSTAIN, 3) == {"abstained": True, "sentences": []}


def test_oracle_inserts_the_gold_chunk_only_when_missing():
    assert oracle_context(QUESTION, CHUNKS[1:], CHUNKS)[-1]["id"] == "a#0"
    assert oracle_context(QUESTION, CHUNKS, CHUNKS) == CHUNKS


def test_prompt_numbers_the_sources():
    msgs = build_messages("Why?", CHUNKS[:2])
    assert "[1] (A > Symptoms)" in msgs[1]["content"] and "[2] (B > Control)" in msgs[1]["content"]


def test_check_answer_measures_support_and_gold_citation():
    row = {"id": "g1", "type": "grower", "split": "test", "context": ["a#0", "b#0"],
           "answer": "Spots have concentric rings [1]. Spots are purple [1]. "
                     "Remove infected debris [1]."}
    res = check_answer(row, QUESTION, CHUNKS, word_overlap_nli)
    assert res["n_sentences"] == 3 and res["cited_sentences"] == 3
    # 'purple' is not in the source; the debris sentence cites the wrong source
    assert res["supported_sentences"] == 1
    assert res["cites_gold"] and res["gold_in_context"]
    s = summarize([res])
    assert s["supported_share_of_cited"] == 1 / 3 and s["abstention"] == 0


def test_nli_sanity_check_contrasts_own_and_random_chunks():
    res = sanity_check([QUESTION], CHUNKS, word_overlap_nli)
    assert res == {"pairs": 1, "own_chunk_entailed": 1.0, "random_chunk_entailed": 0.0}
