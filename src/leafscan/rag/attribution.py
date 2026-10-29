"""Check generated answers: abstention, citations, and NLI support of each sentence.

    python -m leafscan.rag.attribution     # reads reports/rag/answers_*.jsonl

A sentence is supported when an NLI model finds it entailed by at least one of the chunks it
cites. Before trusting the NLI model, a sanity check: gold quotes must be entailed by their own
chunk and not by a random other chunk.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from leafscan.rag.corpus import chunk_pages, load_pages
from leafscan.rag.evalset import load_questions, relevant_chunks
from leafscan.rag.generate import parse

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
THRESHOLD = 0.5
REPORT_PATH = Path("reports/rag/generation.json")


class NLI:
    """P(entailment) of (premise, hypothesis) pairs."""

    def __init__(self, model_name: str = NLI_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device, self.torch = device, torch
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    def __call__(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), batch_size):
            premises, hyps = zip(*pairs[i:i + batch_size], strict=True)
            x = self.tok(list(premises), list(hyps), truncation="only_first", max_length=512,
                         padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**x).logits.softmax(-1)[:, self.entail]
            out.extend(probs.cpu().tolist())
        return np.array(out)


def sanity_check(questions: list[dict], chunks: list[dict],
                 nli: Callable[[list[tuple[str, str]]], np.ndarray], seed: int = 0) -> dict:
    """Entailment of each gold quote by its own chunk vs by a random chunk of another page."""
    rng = random.Random(seed)
    by_id = {c["id"]: c for c in chunks}
    own, other = [], []
    for q in questions:
        for quote in q.get("evidence", []):
            hits = relevant_chunks({"evidence": [quote]}, chunks)
            if not hits:
                continue
            chunk = by_id[hits[0]]
            own.append((chunk["text"], quote))
            other.append((rng.choice([c for c in chunks if c["page"] != chunk["page"]])["text"],
                          quote))
    p_own, p_other = nli(own), nli(other)
    return {"pairs": len(own),
            "own_chunk_entailed": float((p_own > THRESHOLD).mean()),
            "random_chunk_entailed": float((p_other > THRESHOLD).mean())}


def check_answer(row: dict, question: dict, chunks: list[dict],
                 nli: Callable[[list[tuple[str, str]]], np.ndarray]) -> dict:
    by_id = {c["id"]: c for c in chunks}
    context = [by_id[c] for c in row["context"]]
    parsed = parse(row["answer"], len(context))
    sentences = parsed["sentences"]
    pairs, owner = [], []
    for i, s in enumerate(sentences):
        for n in s["cited"] if s["valid"] else []:
            pairs.append((context[n - 1]["text"], s["text"]))
            owner.append(i)
    probs = nli(pairs) if pairs else np.array([])
    support = [0.0] * len(sentences)
    for i, p in zip(owner, probs, strict=True):
        support[i] = max(support[i], float(p))
    gold = set(relevant_chunks(question, chunks))
    cited_ids = {context[n - 1]["id"] for s in sentences if s["valid"] for n in s["cited"]}
    return {"id": row["id"], "type": row["type"], "split": row["split"],
            "abstained": parsed["abstained"], "n_sentences": len(sentences),
            "cited_sentences": sum(s["valid"] for s in sentences),
            "supported_sentences": sum(p > THRESHOLD for p in support),
            "gold_in_context": bool(gold & set(row["context"])),
            "cites_gold": bool(gold & cited_ids), "support": support}


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if not r["abstained"]]
    sentences = sum(r["n_sentences"] for r in answered)
    cited = sum(r["cited_sentences"] for r in answered)
    supported = sum(r["supported_sentences"] for r in answered)
    out = {"n": len(rows), "abstention": float(np.mean([r["abstained"] for r in rows])),
           "sentences": sentences,
           "cited_share": cited / sentences if sentences else None,
           "supported_share_of_cited": supported / cited if cited else None,
           "supported_share_of_all": supported / sentences if sentences else None}
    if rows and rows[0]["type"] != "unanswerable":
        out["cites_gold_when_answered"] = (float(np.mean([r["cites_gold"] for r in answered]))
                                           if answered else None)
        out["gold_in_context"] = float(np.mean([r["gold_in_context"] for r in rows]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="*", default=["retrieved", "oracle"])
    args = ap.parse_args()

    chunks = chunk_pages(load_pages(), 200)
    questions = {q["id"]: q for q in load_questions()}
    nli = NLI()
    report = {"nli_model": NLI_MODEL, "threshold": THRESHOLD,
              "nli_sanity": sanity_check(list(questions.values()), chunks, nli),
              "results": {}, "per_answer": {}}
    print("NLI sanity:", report["nli_sanity"], flush=True)
    for cond in args.conditions:
        path = Path(f"reports/rag/answers_{cond}.jsonl")
        answers = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [check_answer(a, questions[a["id"]], chunks, nli) for a in answers]
        report["per_answer"][cond] = rows
        report["results"][cond] = {
            split: {t: summarize([r for r in rows if r["type"] == t
                                  and split in ("all", r["split"])])
                    for t in ("extractive", "grower", "unanswerable")}
            for split in ("dev", "test", "all")}
        report["results"][cond]["seconds_per_answer"] = float(np.median([a["seconds"]
                                                                         for a in answers]))
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["results"], indent=1))


if __name__ == "__main__":
    main()
