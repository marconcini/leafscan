"""Use the photo's diagnosis in retrieval (H4; plan: research log, 2026-10-29).

    python -m leafscan.rag.diagnosis predict    # classifier on PlantDoc test photos
    python -m leafscan.rag.diagnosis evaluate   # recall@5 of each condition, dev and test

A (question, photo) pair stands for: a grower photographs a leaf of the question's disease, then
asks the question. The classifier's real output on that photo decides what retrieval sees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leafscan.data import pretty_label
from leafscan.rag.evalset import load_questions, relevant_chunks
from leafscan.rag.retrieval_eval import bootstrap_ci

PREDICTIONS_PATH = Path("reports/rag/field_predictions.json")
REPORT_PATH = Path("reports/rag/diagnosis.json")
CONDITIONS = ("none", "oracle", "top-1", "adaptive", "prefix")
K = 5


def allowed_classes(condition: str, top: list[tuple[str, float]], confident: bool,
                    true_class: str | None = None) -> list[str] | None:
    """Classes whose pages retrieval may use; None means no filter."""
    if condition == "oracle":
        return [true_class]
    if condition == "top-1" or (condition == "adaptive" and confident):
        return [top[0][0]]
    if condition == "adaptive":
        return [c for c, _ in top[:2]]
    return None


def query_for(question: str, top: list[tuple[str, float]], prefix: bool) -> str:
    return f"{pretty_label(top[0][0])}: {question}" if prefix else question


def search(retriever, chunks: list[dict], query: str, classes: list[str] | None,
           k: int = K) -> list[tuple[str, float]]:
    """Top-k chunks, restricted to pages of `classes` (all chunks if none match)."""
    ranked = retriever.search(query, len(chunks))
    if classes is None:
        return ranked[:k]
    keep = {c["id"] for c in chunks if set(c["classes"]) & set(classes)}
    return [r for r in ranked if r[0] in keep][:k] if keep else ranked[:k]


def retrieve(condition: str, retriever, chunks: list[dict], question: str,
             top: list[tuple[str, float]], confident: bool,
             true_class: str | None = None) -> list[tuple[str, float]]:
    return search(retriever, chunks, query_for(question, top, condition == "prefix"),
                  allowed_classes(condition, top, confident, true_class))


def predict_field(model: str) -> list[dict]:
    """The deployed classifier's calibrated top-3 on every PlantDoc test photo."""
    from PIL import Image

    from leafscan.field import load_plantdoc
    from leafscan.predictor import Predictor

    predictor = Predictor(model)
    items, _ = load_plantdoc()
    out = []
    for item in (i for i in items if i["split"] == "test"):
        with Image.open(item["path"]) as im:
            pred = predictor(im, explain=False)
        out.append({"photo": f"{item['path'].parent.name}/{item['path'].name}",
                    "label": item["label"], "top": pred.top, "confident": pred.confident})
    return out


def evaluate(questions: list[dict], photos: list[dict], chunks: list[dict], retriever) -> dict:
    """Per question and condition: recall@5 averaged over the photos of its disease."""
    per_question = {}
    for q in questions:
        gold = set(relevant_chunks(q, chunks))
        mine = [p for p in photos if p["label"] == q["classes"][0]]
        if not gold or not mine:
            continue
        hits = {cond: [] for cond in CONDITIONS}
        for p in mine:
            top = [tuple(t) for t in p["top"]]
            for cond in CONDITIONS:
                ids = {cid for cid, _ in retrieve(cond, retriever, chunks, q["question"], top,
                                                  p["confident"], q["classes"][0])}
                hits[cond].append(bool(ids & gold))
        per_question[q["id"]] = {
            "type": q["type"], "split": q["split"], "photos": len(mine),
            "top1_right": float(np.mean([p["top"][0][0] == p["label"] for p in mine])),
            **{cond: float(np.mean(h)) for cond, h in hits.items()}}
    return per_question


def summarize(per_question: dict, split: str) -> dict:
    out = {}
    for t in ("all", "extractive", "grower"):
        rows = [r for r in per_question.values()
                if r["split"] == split and t in ("all", r["type"])]
        out[t] = {"n_questions": len(rows), "n_pairs": sum(r["photos"] for r in rows),
                  "top1_right": float(np.mean([r["top1_right"] for r in rows]))}
        for cond in CONDITIONS:
            v = np.array([r[cond] for r in rows])
            out[t][cond] = {"recall@5": float(v.mean()), "ci95": bootstrap_ci(v)}
            if cond != "none":   # paired difference with the current system
                d = v - np.array([r["none"] for r in rows])
                out[t][cond]["diff_vs_none"] = float(d.mean())
                out[t][cond]["diff_ci95"] = bootstrap_ci(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["predict", "evaluate"])
    ap.add_argument("--model", default="models/leafscan_field/best.pt")
    args = ap.parse_args()

    if args.command == "predict":
        preds = predict_field(args.model)
        PREDICTIONS_PATH.write_text(json.dumps(preds, indent=1))
        print(len(preds), "photos, top-1 accuracy",
              round(np.mean([p["top"][0][0] == p["label"] for p in preds]), 3))
        return

    from leafscan.rag.corpus import chunk_pages, load_pages
    from leafscan.rag.generate import RETRIEVER
    from leafscan.rag.retrieval import Dense

    chunks = chunk_pages(load_pages(), 200)
    questions = [q for q in load_questions() if q["type"] != "unanswerable"]
    photos = json.loads(PREDICTIONS_PATH.read_text())
    per_question = evaluate(questions, photos, chunks, Dense(chunks, RETRIEVER))
    report = {"k": K, "results": {s: summarize(per_question, s) for s in ("dev", "test")},
              "per_question": per_question}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    for split in ("dev", "test"):
        for t, m in report["results"][split].items():
            cells = " | ".join(f"{m[c]['recall@5']:.2f}" for c in CONDITIONS)
            print(f"| {split} | {t} | {m['n_questions']} | {m['top1_right']:.2f} | {cells} |")


if __name__ == "__main__":
    main()
