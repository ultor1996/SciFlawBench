"""A small, deterministic BM25 ranker."""

from __future__ import annotations

import math
import re
from collections import Counter

from .canon import fold

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    fold(word)
    for word in [
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "was",
        "are",
        "be",
        "an",
        "a",
        "it",
        "its",
        "this",
        "that",
    ]
)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN.findall(fold(text or ""))
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """BM25 over `(doc_id, title, text)` documents; the title is weighted `title_weight`x."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, title_weight: int = 3):
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        self._ids: list[str] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._postings: dict[str, list[int]] = {}
        self._avgdl = 0.0

    def add(self, doc_id: str, title: str = "", text: str = "") -> None:
        tokens = tokenize(title) * self.title_weight + tokenize(text)
        tf = Counter(tokens)
        index = len(self._ids)
        self._ids.append(doc_id)
        self._tf.append(tf)
        self._len.append(len(tokens))
        for term in tf:
            self._df[term] += 1
            self._postings.setdefault(term, []).append(index)

    def finalize(self) -> BM25Index:
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        return self

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        candidates: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top `limit` `(doc_id, score)` pairs with score > `min_score`.

        `candidates`, when given, restricts the ranking to those doc ids (used for
        fielded/boolean arXiv queries).
        """
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self._ids:
            return []
        n = len(self._ids)
        scores: dict[int, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i in postings:
                if candidates is not None and self._ids[i] not in candidates:
                    continue
                tf = self._tf[i][term]
                norm = 1.0 - self.b + self.b * (self._len[i] / self._avgdl if self._avgdl else 0)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        ranked = sorted(
            ((self._ids[i], round(s, 9)) for i, s in scores.items() if round(s, 9) > min_score),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[: max(0, limit)]
