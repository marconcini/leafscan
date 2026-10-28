"""Evaluate retrievers on the evaluation set.

    python -m leafscan.rag.retrieval_eval [--max-words 200]

Writes reports/rag/retrieval_baselines.json (per-question ranks included) and prints a
markdown table for the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from leafscan.rag.corpus import chunk_pages, load_pages
from leafscan.rag.evalset import load_questions, relevant_chunks
from leafscan.rag.retrieval import BM25, Dense

REPORT_PATH = Path("reports/rag/retrieval_baselines.json")
DENSE_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-m3"]
KS = (1, 5, 10)


def run(retriever, questions: list[dict], chunks: list[dict], k: int = 10) -> list[dict]:
    """Per question: rank (1-based) of the first relevant chunk in the top k, and top-1 score."""
    out = []
    for q in questions:
        results = retriever.search(q["question"], k)
        gold = set(relevant_chunks(q, chunks))
        rank = next((i for i, (cid, _) in enumerate(results, 1) if cid in gold), None)
        out.append({"id": q["id"], "type": q["type"], "split": q["split"], "rank": rank,
                    "top1_score": results[0][1]})
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metrics(rows: list[dict]) -> dict:
    """Recall@k and MRR@10 of answerable rows; AUROC of the top-1 score vs answerability."""
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    out = {"n": len(answerable)}
    if answerable:
        ranks = [r["rank"] for r in answerable]
        for k in KS:
            hits = np.array([rk is not None and rk <= k for rk in ranks], dtype=float)
            out[f"recall@{k}"] = float(hits.mean())
            if k == 5:
                out["recall@5_ci"] = bootstrap_ci(hits)
        out["mrr@10"] = float(np.mean([1 / rk if rk else 0.0 for rk in ranks]))
    labels = [r["type"] != "unanswerable" for r in rows]
    if 0 < sum(labels) < len(labels):
        out["answerable_auroc"] = float(roc_auc_score(labels, [r["top1_score"] for r in rows]))
    return out


def by_type(rows: list[dict], split: str) -> dict:
    rows = [r for r in rows if r["split"] == split]
    out = {"all": metrics(rows)}
    for t in sorted({r["type"] for r in rows} - {"unanswerable"}):
        out[t] = metrics([r for r in rows if r["type"] == t])
    return out


def mcnemar_hits(a: list[dict], b: list[dict], k: int = 5) -> dict:
    """Paired test on 'relevant chunk in top k' for the same answerable questions."""
    hit = {r["id"]: r["rank"] is not None and r["rank"] <= k for r in a}
    only_a = sum(hit[r["id"]] and not (r["rank"] and r["rank"] <= k) for r in b)
    only_b = sum(not hit[r["id"]] and bool(r["rank"] and r["rank"] <= k) for r in b)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--models", nargs="*", default=DENSE_MODELS)
    args = ap.parse_args()

    chunks = chunk_pages(load_pages(), args.max_words)
    questions = load_questions()
    retrievers = [BM25(chunks)] + [Dense(chunks, m) for m in args.models]
    rows = {r.name: run(r, questions, chunks) for r in retrievers}
    report = {"max_words": args.max_words, "chunks": len(chunks),
              "results": {name: {s: by_type(rs, s) for s in ("dev", "test")}
                          for name, rs in rows.items()},
              "mcnemar_recall@5_vs_bm25": {}, "per_question": rows}
    for name, rs in rows.items():
        if name != "bm25":
            report["mcnemar_recall@5_vs_bm25"][name] = {
                split: {t: mcnemar_hits([r for r in rows["bm25"] if r["split"] == split
                                         and (t == "all" or r["type"] == t)
                                         and r["type"] != "unanswerable"],
                                        [r for r in rs if r["split"] == split
                                         and (t == "all" or r["type"] == t)
                                         and r["type"] != "unanswerable"])
                        for t in ("all", "extractive", "grower")}
                for split in ("dev", "test")}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("| Retriever | Type | n | R@1 | R@5 (95% CI) | R@10 | MRR@10 |")
    print("|---|---|---:|---:|---|---:|---:|")
    for name, res in report["results"].items():
        for t, m in res["test"].items():
            lo, hi = m["recall@5_ci"]
            print(f"| {name} | {t} | {m['n']} | {m['recall@1']:.2f} | {m['recall@5']:.2f} "
                  f"({lo:.2f}–{hi:.2f}) | {m['recall@10']:.2f} | {m['mrr@10']:.2f} |")
    for name, res in report["results"].items():
        print(name, "answerable AUROC (test):", round(res["test"]["all"]["answerable_auroc"], 3))
    print(json.dumps(report["mcnemar_recall@5_vs_bm25"], indent=1))


if __name__ == "__main__":
    main()
