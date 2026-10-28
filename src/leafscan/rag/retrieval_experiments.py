"""Retrieval experiments (plan: research log, 2026-10-28): stemming, RRF, chunk size, reranking.

    python -m leafscan.rag.retrieval_experiments --split dev
    python -m leafscan.rag.retrieval_experiments --split test --only "bge-small" "<chosen>"

Besides recall@k, reports recall within a budget of retrieved words, so that chunk sizes are
compared at equal prompt length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leafscan.rag.corpus import chunk_pages, load_pages
from leafscan.rag.evalset import load_questions, relevant_chunks
from leafscan.rag.retrieval import BM25, RRF, Dense, Rerank
from leafscan.rag.retrieval_eval import bootstrap_ci, mcnemar_hits

DENSE = "BAAI/bge-small-en-v1.5"
RERANKERS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]
SIZES = (100, 200, 400)
DEPTH = 20
WORD_BUDGET = 1000


def evaluate(retriever, questions: list[dict], chunks: list[dict]) -> list[dict]:
    """Rank of the first relevant chunk, and the words read up to and including it."""
    length = {c["id"]: len(c["text"].split()) for c in chunks}
    rows = []
    for q in questions:
        ranked = [cid for cid, _ in retriever.search(q["question"], DEPTH)]
        gold = set(relevant_chunks(q, chunks))
        rank = next((i for i, cid in enumerate(ranked, 1) if cid in gold), None)
        words_read = sum(length[c] for c in ranked[:rank]) if rank else None
        rows.append({"id": q["id"], "type": q["type"], "rank": rank, "words_read": words_read})
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    for t in ("all", "extractive", "grower"):
        rs = [r for r in rows if r["type"] != "unanswerable" and t in ("all", r["type"])]
        hit5 = np.array([bool(r["rank"] and r["rank"] <= 5) for r in rs], dtype=float)
        # the first chunk is always read, whatever its length
        budget = np.array([bool(r["rank"] and (r["rank"] == 1 or r["words_read"] <= WORD_BUDGET))
                           for r in rs], dtype=float)
        out[t] = {"n": len(rs), "recall@1": float(np.mean([r["rank"] == 1 for r in rs])),
                  "recall@5": float(hit5.mean()), "recall@5_ci": bootstrap_ci(hit5),
                  f"recall@{WORD_BUDGET}w": float(budget.mean()),
                  "mrr@20": float(np.mean([1 / r["rank"] if r["rank"] else 0 for r in rs]))}
    return out


def configurations(chunks: list[dict], size: int):
    """The retrievers of the plan, built lazily so that --only skips the expensive ones."""
    cache = {}

    def get(name, make):
        if name not in cache:
            cache[name] = make()
        return cache[name]

    bm25 = lambda: get("bm25", lambda: BM25(chunks))  # noqa: E731
    stem = lambda: get("stem", lambda: BM25(chunks, stem=True))  # noqa: E731
    dense = lambda: get("dense", lambda: Dense(chunks, DENSE))  # noqa: E731
    configs = {"bm25": bm25, "bm25+stem": stem, "bge-small": dense,
               "rrf(bm25, bge-small)": lambda: RRF([bm25(), dense()]),
               "rrf(bm25+stem, bge-small)": lambda: RRF([stem(), dense()])}
    if size == 200:   # reranking is tested at the baseline chunk size only
        for m in RERANKERS:
            short = m.split("/")[-1]
            configs[f"bge-small > {short}"] = lambda m=m: Rerank(dense(), chunks, m, DEPTH)
            configs[f"rrf(bm25+stem, bge-small) > {short}"] = (
                lambda m=m: Rerank(RRF([stem(), dense()]), chunks, m, DEPTH))
    return configs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    ap.add_argument("--only", nargs="*", help="configuration names to run (default: all)")
    args = ap.parse_args()

    questions = [q for q in load_questions() if q["split"] == args.split]
    pages = load_pages()
    results, rows = {}, {}
    for size in args.sizes:
        chunks = chunk_pages(pages, size)
        for name, make in configurations(chunks, size).items():
            if args.only and name not in args.only:
                continue
            key = f"{name} @ {size}w"
            rows[key] = evaluate(make(), questions, chunks)
            results[key] = summary(rows[key])
            g, e = results[key]["grower"], results[key]["extractive"]
            print(f"| {key} | {e['recall@5']:.2f} | {g['recall@5']:.2f} | "
                  f"{results[key]['all']['recall@5']:.2f} | "
                  f"{results[key]['all'][f'recall@{WORD_BUDGET}w']:.2f} | "
                  f"{results[key]['all']['mrr@20']:.2f} |", flush=True)

    base = f"bge-small @ {200}w"
    paired = {}
    if base in rows:
        answerable = lambda rs: [r for r in rs if r["type"] != "unanswerable"]  # noqa: E731
        paired = {k: mcnemar_hits(answerable(rows[base]), answerable(v))
                  for k, v in rows.items() if k != base}
    out = Path(f"reports/rag/retrieval_experiments_{args.split}.json")
    out.write_text(json.dumps({"split": args.split, "depth": DEPTH, "word_budget": WORD_BUDGET,
                               "results": results, "mcnemar_recall@5_vs_bge_small": paired,
                               "per_question": rows}, indent=2))


if __name__ == "__main__":
    main()
