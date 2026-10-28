"""Build the RAG corpus: English Wikipedia articles on the 26 PlantVillage diseases.

    python -m leafscan.rag.corpus fetch     # downloads the articles -> corpus/wikipedia.jsonl
    python -m leafscan.rag.corpus stats     # words, chunks and topics covered per class

Wikipedia text is CC BY-SA 4.0 (compatible with the CC BY-SA weights); each page keeps its
revision id and URL for attribution. The snapshot is versioned so experiments are reproducible
even when the articles change.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

CORPUS_PATH = Path("corpus/wikipedia.jsonl")
REPORT_PATH = Path("reports/rag/corpus.json")
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "LeafScan-research/0.1 (https://github.com/marconcini/leafscan)"

# Wikipedia article -> PlantVillage classes it documents. A page can cover several classes
# (one pathogen on two crops) and a class can have several pages (disease + pathogen).
PAGES = {
    "Apple scab": ["Apple___Apple_scab"],
    "Venturia inaequalis": ["Apple___Apple_scab"],
    "Botryosphaeria obtusa": ["Apple___Black_rot"],
    "Gymnosporangium juniperi-virginianae": ["Apple___Cedar_apple_rust"],
    "Podosphaera clandestina": ["Cherry_(including_sour)___Powdery_mildew"],
    "Erysiphaceae": ["Cherry_(including_sour)___Powdery_mildew", "Squash___Powdery_mildew"],
    "Podosphaera xanthii": ["Squash___Powdery_mildew"],
    "Corn grey leaf spot": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"],
    "Puccinia sorghi": ["Corn_(maize)___Common_rust_"],
    "Northern corn leaf blight": ["Corn_(maize)___Northern_Leaf_Blight"],
    "Setosphaeria turcica": ["Corn_(maize)___Northern_Leaf_Blight"],
    "Black rot (grape disease)": ["Grape___Black_rot"],
    "Esca (grape disease)": ["Grape___Esca_(Black_Measles)"],
    "Grapevine trunk disease": ["Grape___Esca_(Black_Measles)"],
    "Pseudocercospora vitis": ["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"],
    "Citrus greening disease": ["Orange___Haunglongbing_(Citrus_greening)"],
    "Xanthomonas arboricola pv. pruni": ["Peach___Bacterial_spot"],
    "Bacterial leaf spot of peppers and tomato": ["Pepper,_bell___Bacterial_spot",
                                                  "Tomato___Bacterial_spot"],
    "Alternaria solani": ["Potato___Early_blight", "Tomato___Early_blight"],
    "Phytophthora infestans": ["Potato___Late_blight", "Tomato___Late_blight"],
    "Diplocarpon earlianum": ["Strawberry___Leaf_scorch"],
    "Tomato leaf mold": ["Tomato___Leaf_Mold"],
    "Passalora fulva": ["Tomato___Leaf_Mold"],
    "Septoria lycopersici": ["Tomato___Septoria_leaf_spot"],
    "Tetranychus urticae": ["Tomato___Spider_mites Two-spotted_spider_mite"],
    "Corynespora cassiicola": ["Tomato___Target_Spot"],
    "Tomato yellow leaf curl virus": ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"],
    "Tomato mosaic virus": ["Tomato___Tomato_mosaic_virus"],
}


def _api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_page(title: str) -> dict:
    """Plain text of the current revision, with '== Section ==' headings kept."""
    d = _api({"action": "query", "titles": title, "redirects": 1,
              "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
              "rvprop": "ids|timestamp", "inprop": "url"})
    page = d["query"]["pages"][0]
    if page.get("missing"):
        raise KeyError(f"Wikipedia page not found: {title}")
    rev = page["revisions"][0]
    return {"title": page["title"], "url": page["fullurl"], "revid": rev["revid"],
            "timestamp": rev["timestamp"], "license": "CC BY-SA 4.0",
            "classes": PAGES[title], "text": page["extract"]}


def fetch(out: Path = CORPUS_PATH, delay: float = 1.0) -> list[dict]:
    pages = []
    for title in PAGES:
        pages.append(fetch_page(title))
        print(f"{title}: {len(pages[-1]['text'])} chars (rev {pages[-1]['revid']})")
        time.sleep(delay)   # be polite to the API
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages))
    return pages


def load_pages(path: Path = CORPUS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Sections without content to retrieve (their subsections are dropped too).
DROP_SECTIONS = {"references", "external links", "see also", "further reading", "notes",
                 "gallery", "bibliography", "sources", "citations", "footnotes"}
HEADING = re.compile(r"^(=+)\s*(.*?)\s*=+$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# words that mark a section about what a grower can do
MANAGEMENT = re.compile(r"management|control|treatment|resistance|prevention", re.I)


def sections(page: dict) -> list[dict]:
    """Split a page into sections; 'path' is [page title, heading, subheading, ...]."""
    out, path, dropped_level = [], [page["title"]], None
    current = {"path": list(path), "lines": []}
    for line in page["text"].splitlines():
        m = HEADING.match(line.strip())
        if not m:
            if dropped_level is None:
                current["lines"].append(line)
            continue
        level, name = len(m.group(1)) - 1, m.group(2)
        out.append(current)
        if dropped_level is not None and level > dropped_level:
            continue                       # subsection of a dropped section
        dropped_level = level if name.lower() in DROP_SECTIONS else None
        path = path[:level] + [name]
        current = {"path": list(path), "lines": []}
    out.append(current)
    return [{"path": sec["path"], "text": "\n".join(sec["lines"]).strip()}
            for sec in out if "\n".join(sec["lines"]).strip()]


def _pieces(paragraph: str, max_words: int) -> list[str]:
    """A paragraph, or its sentences when it is longer than max_words."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    return SENTENCE_END.split(paragraph)


def chunk_text(text: str, max_words: int) -> list[str]:
    """Greedy packing of paragraphs (or sentences) into chunks of at most max_words.

    A single sentence longer than max_words becomes its own, longer chunk.
    """
    chunks, current, n = [], [], 0
    for paragraph in re.split(r"\n\s*\n", text):
        for piece in _pieces(" ".join(paragraph.split()), max_words):
            words = len(piece.split())
            if not words:
                continue
            if current and n + words > max_words:
                chunks.append(" ".join(current))
                current, n = [], 0
            current.append(piece)
            n += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[dict], max_words: int = 200) -> list[dict]:
    """Chunks never cross a section boundary; each keeps its page, section path and classes."""
    chunks = []
    for page in pages:
        i = 0
        for sec in sections(page):
            for text in chunk_text(sec["text"], max_words):
                chunks.append({"id": f"{page['title']}#{i}", "page": page["title"],
                               "section": " > ".join(sec["path"]), "text": text,
                               "classes": page["classes"], "url": page["url"]})
                i += 1
    return chunks


def stats(pages: list[dict], max_words: int = 200) -> dict:
    """Corpus size per class, and whether each class has a management section."""
    chunks = chunk_pages(pages, max_words)
    per_class = defaultdict(lambda: {"pages": 0, "words": 0, "chunks": 0,
                                     "has_management_section": False})
    for page in pages:
        for c in page["classes"]:
            per_class[c]["pages"] += 1
    for ch in chunks:
        for c in ch["classes"]:
            per_class[c]["words"] += len(ch["text"].split())
            per_class[c]["chunks"] += 1
            if MANAGEMENT.search(ch["section"].split(" > ", 1)[-1]):
                per_class[c]["has_management_section"] = True
    words = [len(ch["text"].split()) for ch in chunks]
    return {"pages": len(pages), "chunks": len(chunks), "max_words": max_words,
            "words": sum(words), "chunk_words_median": sorted(words)[len(words) // 2],
            "per_class": dict(sorted(per_class.items()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "stats"])
    ap.add_argument("--max-words", type=int, default=200)
    args = ap.parse_args()
    if args.command == "fetch":
        fetch()
    else:
        res = stats(load_pages(), args.max_words)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(res, indent=2))
        print(f"{res['pages']} pages, {res['chunks']} chunks, {res['words']} words")
        print("| Class | Pages | Words | Chunks | Management section |\n|---|---:|---:|---:|---|")
        for c, v in res["per_class"].items():
            print(f"| {c} | {v['pages']} | {v['words']} | {v['chunks']} | "
                  f"{'yes' if v['has_management_section'] else 'no'} |")


if __name__ == "__main__":
    main()
