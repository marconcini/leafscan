"""Retrievers over the chunked corpus: BM25 (lexical) and dense embeddings.

Both index the same text: the chunk's section path followed by its content, so that a
"Management" chunk still carries the name of the disease its page is about.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np

from leafscan.rag.evalset import STOPWORDS, words


def indexed_text(chunk: dict) -> str:
    return f"{chunk['section']}\n{chunk['text']}"


def tokenize(text: str, stem: bool = False) -> list[str]:
    tokens = [w for w in words(text) if w not in STOPWORDS]
    if stem:
        import snowballstemmer

        tokens = snowballstemmer.stemmer("english").stemWords(tokens)
    return tokens


class BM25:
    """Okapi BM25 [Robertson & Zaragoza 2009], with the non-negative idf used by Lucene."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75, stem: bool = False):
        self.name = "bm25+stem" if stem else "bm25"
        self.ids, self.stem = [c["id"] for c in chunks], stem
        self.docs = [Counter(tokenize(indexed_text(c), stem)) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for t in set(tokenize(query, self.stem)):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=float)
            out += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return top_k(self.ids, self.scores(query), k)


# Instructions some embedding models were trained with (see their model cards).
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5":
                "Represent this sentence for searching relevant passages: ",
                "intfloat/e5-base-v2": "query: "}
PASSAGE_PREFIX = {"intfloat/e5-base-v2": "passage: "}


class Dense:
    """Cosine similarity between normalised embeddings of the query and of each chunk.

    `encode` maps a list of texts to an array of embeddings; by default a
    sentence-transformers model is loaded (tests pass a toy encoder instead).
    """

    def __init__(self, chunks: list[dict], model_name: str,
                 encode: Callable[[list[str]], np.ndarray] | None = None):
        self.name, self.ids = model_name, [c["id"] for c in chunks]
        if encode is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device="cpu")

            def encode(texts):
                return model.encode(texts, batch_size=16, normalize_embeddings=True)
        self.encode = encode
        prefix = PASSAGE_PREFIX.get(model_name, "")
        self.matrix = self._normalise(self.encode([prefix + indexed_text(c) for c in chunks]))

    @staticmethod
    def _normalise(x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)

    def scores(self, query: str) -> np.ndarray:
        q = self._normalise(self.encode([QUERY_PREFIX.get(self.name, "") + query]))[0]
        return self.matrix @ q

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return top_k(self.ids, self.scores(query), k)


def top_k(ids: list[str], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [(ids[i], float(scores[i])) for i in order]


class RRF:
    """Reciprocal rank fusion [Cormack et al. 2009]: score = sum over retrievers of 1/(k + rank)."""

    def __init__(self, retrievers: list, k: int = 60, depth: int = 100):
        self.retrievers, self.k, self.depth = retrievers, k, depth
        self.name = "rrf(" + " + ".join(r.name for r in retrievers) + ")"

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for r in self.retrievers:
            for rank, (cid, _) in enumerate(r.search(query, self.depth), 1):
                fused[cid] = fused.get(cid, 0.0) + 1 / (self.k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


class Rerank:
    """Re-score the first stage's top `depth` chunks with a cross-encoder [Nogueira & Cho 2019].

    `score_pairs` maps (query, passage) pairs to relevance scores; by default a
    sentence-transformers CrossEncoder is loaded.
    """

    def __init__(self, first_stage, chunks: list[dict], model_name: str, depth: int = 20,
                 score_pairs: Callable[[list[tuple[str, str]]], np.ndarray] | None = None):
        self.first, self.depth = first_stage, depth
        self.name = f"{first_stage.name} > {model_name}"
        self.text = {c["id"]: indexed_text(c) for c in chunks}
        if score_pairs is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device="cpu")

            def score_pairs(pairs):
                return model.predict(pairs, batch_size=16)
        self.score_pairs = score_pairs

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        ids = [cid for cid, _ in self.first.search(query, self.depth)]
        scores = np.asarray(self.score_pairs([(query, self.text[c]) for c in ids]), dtype=float)
        return top_k(ids, scores, k)
