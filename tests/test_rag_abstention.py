from leafscan.rag.abstention import apply, evaluate, fit_gate


def row(qid, kind, support, abstained=False):
    return {"id": qid, "type": kind, "abstained": abstained, "n_sentences": len(support),
            "support": support}


def test_policies():
    r = row("q", "grower", [0.9, 0.1])
    assert apply("model", r, top1=0.2, tau=0.5) == {"answered": True, "shown": 2, "removed": 0}
    assert apply("gate", r, top1=0.2, tau=0.5)["answered"] is False
    assert apply("filter", r, top1=0.2, tau=0.5) == {"answered": True, "shown": 1, "removed": 1}
    assert apply("filter", row("u", "unanswerable", [0.0]), 0.9, 0.5)["answered"] is False
    assert apply("model", row("a", "grower", [], abstained=True), 0.9, 0.5)["answered"] is False


def test_gate_is_fitted_on_dev_answerable_only():
    qs = [{"id": f"d{i}", "split": "dev", "type": "grower"} for i in range(11)] + \
         [{"id": "u", "split": "dev", "type": "unanswerable"},
          {"id": "t", "split": "test", "type": "grower"}]
    top1 = {f"d{i}": i / 10 for i in range(11)} | {"u": -5.0, "t": -5.0}
    assert abs(fit_gate(top1, qs) - 0.1) < 1e-9


def test_evaluate_counts_answered_unanswerable_and_removed_sentences():
    rows = [row("a", "grower", [0.9, 0.2]), row("u", "unanswerable", [0.1])]
    res = evaluate(rows, {"a": 0.9, "u": 0.9}, tau=0.5)
    assert res["model"]["unanswerable_answered"] == 1.0
    assert res["filter"]["unanswerable_answered"] == 0.0
    assert res["filter"]["answerable_sentences_removed"] == 0.5
