"""The question-answering assistant of the app: retrieve, generate with citations, check support.

Design choices come from the research log (2026-10-29):
- retrieval uses the question as written; the diagnosis only suggests questions that name it;
- sentences their cited source does not entail are flagged, not removed;
- the model's own "I don't know" is the abstention mechanism (no retrieval gate).
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from leafscan.data import pretty_label
from leafscan.rag.attribution import THRESHOLD
from leafscan.rag.generate import ABSTAIN, TOP_K, build_messages, parse

WARNING = "⚠️ *not found in the cited source*"


def suggested_questions(top: list[tuple[str, float]], confident: bool) -> list[str]:
    """Questions about the predicted disease, or the two most likely ones when unsure."""
    out = []
    for label, _ in top[:1 if confident else 2]:
        crop, _, disease = pretty_label(label).partition(" — ")
        if disease == "healthy":
            continue
        name = f"{disease.lower()} on {crop.lower()}"
        out += [f"What are the symptoms of {name}?", f"How can I manage {name}?"]
    return out


class Assistant:
    def __init__(self, chunks: list[dict], retriever, generate: Callable[[list[dict]], str],
                 nli: Callable[[list[tuple[str, str]]], np.ndarray], k: int = TOP_K):
        self.by_id = {c["id"]: c for c in chunks}
        self.retriever, self.generate, self.nli, self.k = retriever, generate, nli, k

    def __call__(self, question: str) -> dict:
        context = [self.by_id[cid] for cid, _ in self.retriever.search(question, self.k)]
        answer = self.generate(build_messages(question, context))
        parsed = parse(answer, len(context))
        sentences = parsed["sentences"]
        pairs = [(context[n - 1]["text"], s["text"])
                 for s in sentences if s["valid"] for n in s["cited"]]
        probs = iter(self.nli(pairs) if pairs else [])
        for s in sentences:
            support = [float(next(probs)) for _ in s["cited"]] if s["valid"] else []
            s["supported"] = max(support, default=0.0) > THRESHOLD
        return {"question": question, "abstained": parsed["abstained"], "raw": answer,
                "sentences": sentences, "sources": context}


def render(result: dict) -> str:
    """Markdown: the answer with its citations and warnings, then the sources it cites."""
    if result["abstained"]:
        return (f"**{ABSTAIN}**\n\nThe sources retrieved for this question do not answer it. "
                "Try naming the crop and the disease, or ask an expert.")
    lines, cited = [], set()
    for s in result["sentences"]:
        refs = "".join(f"[{n}]" for n in s["cited"])
        flag = "" if s["supported"] else f" {WARNING}"
        text = re.sub(r"\s+([.,;:!?])", r"\1", s["text"])   # "load [2]." -> "load ."
        lines.append(f"- {text} {refs}{flag}")
        if s["valid"]:
            cited.update(s["cited"])
    sources = [f"{n}. [{c['section']}]({c['url']})"
               for n, c in enumerate(result["sources"], 1) if n in cited]
    return ("\n".join(lines) + "\n\n**Sources** (Wikipedia, CC BY-SA 4.0)\n\n"
            + ("\n".join(sources) if sources else "*No valid citation.*"))
