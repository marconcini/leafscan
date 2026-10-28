"""Build the RAG corpus: English Wikipedia articles on the 26 PlantVillage diseases.

    python -m leafscan.rag.corpus fetch     # downloads the articles -> corpus/wikipedia.jsonl

Wikipedia text is CC BY-SA 4.0 (compatible with the CC BY-SA weights); each page keeps its
revision id and URL for attribution. The snapshot is versioned so experiments are reproducible
even when the articles change.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

CORPUS_PATH = Path("corpus/wikipedia.jsonl")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch"])
    args = ap.parse_args()
    if args.command == "fetch":
        fetch()


if __name__ == "__main__":
    main()
