"""Abstention policies on top of the generator's own "I don't know" (plan: research log).

    python -m leafscan.rag.abstention     # needs reports/rag/generation.json

Policies are applied to answers already generated: a retrieval gate (abstain when the best
chunk is not similar enough) and a support filter (drop sentences their sources do not
entail; abstain if none is left).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from leafscan.rag.attribution import THRESHOLD
from leafscan.rag.corpus import chunk_pages, load_pages
from leafscan.rag.evalset import load_questions
from leafscan.rag.generate import RETRIEVER
from leafscan.rag.retrieval import Dense

GATE_QUANTILE = 0.10
REPORT_PATH = Path("reports/rag/abstention.json")
POLICIES = ("model", "gate", "filter", "gate+filter")


def fit_gate(top1: dict[str, float], questions: list[dict]) -> float:
    """τ: the 10th percentile of top-1 scores of dev answerable questions."""
    scores = [top1[q["id"]] for q in questions
              if q["split"] == "dev" and q["type"] != "unanswerable"]
    return float(np.quantile(scores, GATE_QUANTILE))


def apply(policy: str, row: dict, top1: float, tau: float) -> dict:
    """Whether the system answers, and how many sentences it shows / removed."""
    if row["abstained"] or ("gate" in policy and top1 < tau):
        return {"answered": False, "shown": 0, "removed": 0}
    n = row["n_sentences"]
    if "filter" in policy:
        shown = sum(p > THRESHOLD for p in row["support"])
        return {"answered": shown > 0, "shown": shown, "removed": n - shown}
    return {"answered": n > 0, "shown": n, "removed": 0}


def evaluate(rows: list[dict], top1: dict[str, float], tau: float) -> dict:
    out = {}
    for policy in POLICIES:
        res = {r["id"]: apply(policy, r, top1[r["id"]], tau) for r in rows}
        ans = [r for r in rows if r["type"] != "unanswerable"]
        una = [r for r in rows if r["type"] == "unanswerable"]
        kept = sum(res[r["id"]]["shown"] for r in ans)
        removed = sum(res[r["id"]]["removed"] for r in ans)
        out[policy] = {
            "unanswerable_answered": float(np.mean([res[r["id"]]["answered"] for r in una])),
            "answerable_answered": float(np.mean([res[r["id"]]["answered"] for r in ans])),
            "answerable_sentences_removed": removed / (kept + removed) if kept + removed else 0.0,
            "unanswerable_answered_ids": [r["id"] for r in una if res[r["id"]]["answered"]]}
    return out


def main() -> None:
    gen = json.loads(Path("reports/rag/generation.json").read_text())
    rows = gen["per_answer"]["retrieved"]
    questions = load_questions()
    retriever = Dense(chunk_pages(load_pages(), 200), RETRIEVER)
    top1 = {q["id"]: retriever.search(q["question"], 1)[0][1] for q in questions}
    tau = fit_gate(top1, questions)
    report = {"tau": tau, "gate_quantile": GATE_QUANTILE,
              "results": {split: evaluate([r for r in rows if r["split"] == split], top1, tau)
                          for split in ("dev", "test")},
              "top1_score": top1}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"tau = {tau:.3f}")
    for split in ("dev", "test"):
        for policy, m in report["results"][split].items():
            print(f"| {split} | {policy} | {m['unanswerable_answered']:.2f} | "
                  f"{m['answerable_answered']:.2f} | {m['answerable_sentences_removed']:.2f} | "
                  f"{', '.join(m['unanswerable_answered_ids'])} |")


if __name__ == "__main__":
    main()
