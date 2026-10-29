import numpy as np

from leafscan.rag.assistant import WARNING, Assistant, render, suggested_questions
from leafscan.rag.generate import ABSTAIN

CHUNKS = [{"id": f"P#{i}", "page": "P", "section": f"P > S{i}", "url": "https://x",
           "text": f"chunk {i}"} for i in range(3)]


class Retriever:
    def search(self, query, k):
        return [(c["id"], 1.0) for c in CHUNKS[:k]]


def nli(pairs):
    """Entailed only by the second chunk."""
    return np.array([1.0 if premise == "chunk 1" else 0.0 for premise, _ in pairs])


def test_suggestions_name_the_disease_and_cover_two_when_unsure():
    top = [("Tomato___Early_blight", 0.5), ("Tomato___healthy", 0.3), ("Potato___Late_blight", .2)]
    sure = suggested_questions(top, confident=True)
    assert sure == ["What are the symptoms of early blight on tomato?",
                    "How can I manage early blight on tomato?"]
    assert suggested_questions(top, confident=False) == sure          # healthy is skipped
    top[1] = ("Potato___Late_blight", 0.3)
    assert len(suggested_questions(top, confident=False)) == 4


def test_answer_flags_unsupported_sentences_and_lists_cited_sources():
    answer = "It spreads by rain [2]. It loves cold. [1] Burn the leaves. [9]"
    result = Assistant(CHUNKS, Retriever(), lambda m: answer, nli, k=3)("How?")
    assert [s["supported"] for s in result["sentences"]] == [True, False, False]
    md = render(result)
    assert md.count(WARNING) == 2 and "- It spreads by rain. [2]\n" in md
    assert "2. [P > S1](https://x)" in md and "1. [P > S0]" in md and "9." not in md


def test_abstention_is_rendered_without_sources():
    result = Assistant(CHUNKS, Retriever(), lambda m: ABSTAIN, nli)("Can I eat it?")
    assert result["abstained"] and ABSTAIN in render(result) and "Sources" not in render(result)
