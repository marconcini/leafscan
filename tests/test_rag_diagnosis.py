import numpy as np

from leafscan.rag.diagnosis import allowed_classes, evaluate, query_for, search, summarize

CHUNKS = [
    {"id": "Scab#0", "classes": ["Apple___Apple_scab"], "text": "scab lesions on apple leaves"},
    {"id": "Rust#0", "classes": ["Apple___Cedar_apple_rust"], "text": "orange rust spots"},
    {"id": "Blight#0", "classes": ["Tomato___Early_blight"], "text": "brown rings on tomato"},
]


class ToyRetriever:
    """Ranks chunks by the number of query words they contain."""

    def search(self, query, k):
        words = set(query.lower().split())
        scores = [(c["id"], float(len(words & set(c["text"].split())))) for c in CHUNKS]
        return sorted(scores, key=lambda x: -x[1])[:k]


TOP = [("Apple___Cedar_apple_rust", 0.6), ("Apple___Apple_scab", 0.3), ("Tomato___healthy", 0.1)]


def test_allowed_classes():
    assert allowed_classes("none", TOP, True) is None
    assert allowed_classes("prefix", TOP, True) is None
    assert allowed_classes("oracle", TOP, True, "Apple___Apple_scab") == ["Apple___Apple_scab"]
    assert allowed_classes("top-1", TOP, False) == ["Apple___Cedar_apple_rust"]
    assert allowed_classes("adaptive", TOP, True) == ["Apple___Cedar_apple_rust"]
    assert allowed_classes("adaptive", TOP, False) == ["Apple___Cedar_apple_rust",
                                                       "Apple___Apple_scab"]


def test_query_prefix_uses_the_readable_label():
    assert query_for("How to treat it?", TOP, prefix=True) == \
        "Apple — Cedar apple rust: How to treat it?"
    assert query_for("How to treat it?", TOP, prefix=False) == "How to treat it?"


def test_filter_keeps_allowed_pages_and_falls_back_when_none_match():
    r = ToyRetriever()
    assert [c for c, _ in search(r, CHUNKS, "scab lesions", ["Apple___Cedar_apple_rust"], 2)] \
        == ["Rust#0"]
    assert search(r, CHUNKS, "scab lesions", ["Tomato___healthy"], 1)[0][0] == "Scab#0"
    assert search(r, CHUNKS, "scab lesions", None, 1)[0][0] == "Scab#0"


def test_evaluate_averages_over_the_photos_of_the_question_disease():
    q = {"id": "q", "type": "grower", "split": "dev", "classes": ["Apple___Apple_scab"],
         "question": "scab lesions", "evidence": ["scab lesions on apple leaves"]}
    photos = [{"label": "Apple___Apple_scab", "top": [["Apple___Apple_scab", 0.9]],
               "confident": True},
              {"label": "Apple___Apple_scab", "top": TOP, "confident": True},
              {"label": "Tomato___Early_blight", "top": TOP, "confident": True}]
    res = evaluate([q], photos, CHUNKS, ToyRetriever())["q"]
    assert res["photos"] == 2 and res["top1_right"] == 0.5
    assert res["none"] == 1.0 and res["oracle"] == 1.0
    assert res["top-1"] == 0.5          # the wrong diagnosis filters the answer out
    summary = summarize({"q": res}, "dev")["all"]
    assert np.isclose(summary["top-1"]["diff_vs_none"], -0.5)
