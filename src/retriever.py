"""Retrieval over the markdown knowledge base in knowledge/.

Each `## heading` section is one chunk. Chunks carry metadata lines (`key: value`) that the
agent uses as structured facts: synonyms, maps_to, related labels, activity energy ranges.

Ranking is TF-IDF cosine similarity plus a boost when one of a chunk's synonyms appears
verbatim in the request. Pure Python, deterministic, no API key and no embedding model.
"""
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
SYNONYM_BOOST = 1.0
MIN_SCORE = 0.05
_META_RE = re.compile(r"^([a-z_]+):\s*(.+)$")
_TOKEN_RE = re.compile(r"[a-z0-9&]+(?:-[a-z0-9]+)?")


@dataclass
class Chunk:
    source: str              # e.g. "moods.md"
    title: str               # the ## heading, e.g. "sad"
    text: str                # description body
    meta: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    matched: List[str] = field(default_factory=list)   # synonyms found verbatim in the query

    @property
    def kind(self) -> str:
        return self.meta.get("type", "")

    @property
    def synonyms(self) -> List[str]:
        return [s.strip().lower() for s in self.meta.get("synonyms", "").split(",") if s.strip()]

    @property
    def related(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for pair in self.meta.get("related", "").split(","):
            if "=" in pair:
                label, value = pair.split("=", 1)
                out[label.strip().lower()] = float(value)
        return out

    @property
    def cite(self) -> str:
        return f"knowledge/{self.source}#{self.title}"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def phrase_in(phrase: str, text: str) -> bool:
    """Whole-word phrase match, so 'pop' does not match inside 'popular'."""
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def parse_markdown(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    source = os.path.basename(path)
    title: Optional[str] = None
    meta: Dict[str, str] = {}
    body: List[str] = []

    def flush():
        if title is not None:
            chunks.append(Chunk(source, title, " ".join(body).strip(), dict(meta)))

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("## "):
                flush()
                title, meta, body = line[3:].strip().lower(), {}, []
            elif title is None:
                continue
            else:
                m = _META_RE.match(line.strip())
                if m and not body:
                    meta[m.group(1)] = m.group(2).strip()
                elif line.strip():
                    body.append(line.strip())
    flush()
    return chunks


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._docs = [Counter(tokenize(self._index_text(c))) for c in chunks]
        n = len(chunks)
        df: Counter = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self._norms = [self._norm(self._weights(doc)) for doc in self._docs]

    @classmethod
    def from_dir(cls, directory: str = KNOWLEDGE_DIR) -> "Retriever":
        chunks: List[Chunk] = []
        for name in sorted(os.listdir(directory)):
            if name.endswith(".md"):
                chunks.extend(parse_markdown(os.path.join(directory, name)))
        return cls(chunks)

    @staticmethod
    def _index_text(c: Chunk) -> str:
        return " ".join([c.title, c.meta.get("synonyms", ""), c.text])

    def _weights(self, counts: Counter) -> Dict[str, float]:
        return {t: tf * self._idf.get(t, 0.0) for t, tf in counts.items()}

    @staticmethod
    def _norm(weights: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in weights.values())) or 1.0

    def search(self, query: str, k: int = 6, kind: Optional[str] = None) -> List[Chunk]:
        q_text = query.lower()
        q = self._weights(Counter(tokenize(q_text)))
        q_norm = self._norm(q)
        results: List[Chunk] = []
        for chunk, doc, d_norm in zip(self.chunks, self._docs, self._norms):
            if kind and chunk.kind != kind:
                continue
            d = self._weights(doc)
            cosine = sum(w * d.get(t, 0.0) for t, w in q.items()) / (q_norm * d_norm)
            matched = [s for s in chunk.synonyms if phrase_in(s, q_text)]
            score = cosine + (SYNONYM_BOOST if matched else 0.0)
            if score >= MIN_SCORE:
                results.append(Chunk(chunk.source, chunk.title, chunk.text, chunk.meta,
                                     round(score, 4), matched))
        results.sort(key=lambda c: (-c.score, c.source, c.title))
        return results[:k]
