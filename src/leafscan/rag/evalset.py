"""Evaluation set for retrieval: questions with gold evidence quotes.

    python -m leafscan.rag.evalset sample     # passages to write questions from (fixed seed)
    python -m leafscan.rag.evalset check      # validate quotes, overlap and split statistics

Gold evidence is stored as quotes from the articles, not chunk ids, so the same questions
evaluate any chunking: a chunk is relevant when it contains one of the question's quotes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from leafscan.rag.corpus import PAGES, chunk_pages, load_pages

QUESTIONS_PATH = Path("evaluation/questions.jsonl")
REPORT_PATH = Path("reports/rag/evalset.json")
TYPES = ("extractive", "grower", "unanswerable", "human")
DEV_FRACTION = 0.4

# sections a grower's question is likely to be about
GROWER_SECTIONS = re.compile(r"symptom|sign|management|control|treatment|cycle|environment|"
                             r"spread|transmission|prevention|host|identification", re.I)
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
my of on or should so that the their them there these they this to was what when where which
who why will with you your me we our than then also not no into out up""".split())


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS}


def split_of(qid: str) -> str:
    """Deterministic dev/test assignment from the question id."""
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_FRACTION else "test"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for q in qs:
        q.setdefault("split", split_of(q["id"]))
    return qs


def sample_passages(chunks: list[dict], per_class: int = 2, seed: int = 0,
                    min_words: int = 40) -> dict[str, list[dict]]:
    """Passages to write questions from: per disease, `per_class` for each answerable type.

    Grower passages come from symptom / management / spread sections; extractive passages
    from any section, as a synthetic question generator would pick them. A chunk is used once.
    """
    rng = random.Random(seed)
    classes = sorted({c for cs in PAGES.values() for c in cs})
    used, out = set(), {"extractive": [], "grower": []}
    for kind in ("grower", "extractive"):
        for c in classes:
            pool = [ch for ch in chunks if c in ch["classes"] and ch["id"] not in used
                    and len(ch["text"].split()) >= min_words
                    and (kind == "extractive" or GROWER_SECTIONS.search(ch["section"]))]
            for ch in rng.sample(pool, min(per_class, len(pool))):
                used.add(ch["id"])
                out[kind].append({**ch, "for_class": c})
    return out


def locate(quote: str, pages: list[dict]) -> list[str]:
    """Titles of the pages that contain the quote (whitespace and case normalised)."""
    q = normalize(quote)
    return [p["title"] for p in pages if q in normalize(p["text"])]


def relevant_chunks(question: dict, chunks: list[dict]) -> list[str]:
    quotes = [normalize(e) for e in question.get("evidence", [])]
    return [ch["id"] for ch in chunks if any(q in normalize(ch["text"]) for q in quotes)]


def longest_common_ngram(a: str, b: str) -> int:
    """Length of the longest run of consecutive words shared by a and b."""
    wa, wb = words(a), words(b)
    best, prev = 0, [0] * (len(wb) + 1)
    for x in wa:
        cur = [0] * (len(wb) + 1)
        for j, y in enumerate(wb, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def overlap(question: str, evidence: list[str]) -> dict:
    """How much of the question's wording is copied from its evidence."""
    ev = " ".join(evidence)
    q = content_words(question)
    return {"content_word_overlap": len(q & content_words(ev)) / len(q) if q else 0.0,
            "longest_common_ngram": longest_common_ngram(question, ev)}


def check(questions: list[dict], pages: list[dict], max_words: int = 200) -> dict:
    """Validate the set and describe it; raises on any broken gold label."""
    chunks = chunk_pages(pages, max_words)
    ids, problems = Counter(q["id"] for q in questions), []
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    per_type = defaultdict(list)
    for q in questions:
        if q["type"] not in TYPES:
            problems.append(f"{q['id']}: unknown type {q['type']}")
        answerable = q["type"] != "unanswerable"
        if answerable != bool(q.get("evidence")):
            problems.append(f"{q['id']}: evidence must be given iff the question is answerable")
        for e in q.get("evidence", []):
            if not locate(e, pages):
                problems.append(f"{q['id']}: quote not found: {e[:60]}")
        if answerable and not relevant_chunks(q, chunks):
            problems.append(f"{q['id']}: no chunk contains a quote (split across chunks?)")
        per_type[q["type"]].append(q)
    if problems:
        raise ValueError("\n".join(problems))

    summary = {}
    for t, qs in per_type.items():
        row = {"n": len(qs), "dev": sum(split_of(q["id"]) == "dev" for q in qs),
               "classes": len({c for q in qs for c in q.get("classes", [])})}
        if t != "unanswerable":
            ov = [overlap(q["question"], q["evidence"]) for q in qs]
            row["content_word_overlap_mean"] = sum(o["content_word_overlap"] for o in ov) / len(ov)
            row["longest_common_ngram_mean"] = sum(o["longest_common_ngram"] for o in ov) / len(ov)
            row["relevant_chunks_mean"] = sum(len(relevant_chunks(q, chunks))
                                              for q in qs) / len(qs)
        summary[t] = row
    return {"questions": len(questions), "max_words": max_words, "per_type": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "check"])
    args = ap.parse_args()
    pages = load_pages()
    if args.command == "sample":
        for kind, passages in sample_passages(chunk_pages(pages)).items():
            for p in passages:
                print(f"### [{kind}] {p['for_class']} | {p['id']} | {p['section']}\n{p['text']}\n")
    else:
        res = check(load_questions(), pages)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
