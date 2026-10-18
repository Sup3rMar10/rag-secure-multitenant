"""A small BM25 index with an in-index permission pre-filter and exact tombstones."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

from .text import tokenize


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    tenant: str
    text: str
    # Copies of the ACL taken when the chunk was indexed. They go stale when the
    # authoritative ACL changes; the 'baked' retrieval mode trusts them on purpose.
    baked_tenant: str = ""
    baked_groups: frozenset = frozenset()
    tokens: list = field(default_factory=list, repr=False)
    deleted: bool = False


class SearchResult(NamedTuple):
    hits: list          # [(Chunk, score)] best first, already permission-filtered if `allow` was given
    denied: int         # candidates skipped by `allow` (internal use: audit only, never show a caller)
    total_matches: int  # candidates that matched any query term, BEFORE filtering (an existence oracle)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._post: dict = defaultdict(dict)  # term -> {chunk position: tf}
        self._df: dict = defaultdict(int)
        self._n = 0
        self._len_total = 0
        self._by_doc: dict = defaultdict(list)

    # -- building ---------------------------------------------------------
    def add(self, chunk: Chunk) -> None:
        chunk.tokens = tokenize(chunk.text)
        pos = len(self.chunks)
        self.chunks.append(chunk)
        self._by_doc[chunk.doc_id].append(pos)
        counts: dict = defaultdict(int)
        for t in chunk.tokens:
            counts[t] += 1
        for t, tf in counts.items():
            self._post[t][pos] = tf
            self._df[t] += 1
        self._n += 1
        self._len_total += len(chunk.tokens)

    def tombstone_doc(self, doc_id: str) -> int:
        """Remove a document from retrieval AND from the corpus statistics. Returns chunks removed."""
        removed = 0
        for pos in self._by_doc.get(doc_id, []):
            ch = self.chunks[pos]
            if ch.deleted:
                continue
            ch.deleted = True
            removed += 1
            for t in set(ch.tokens):
                self._df[t] -= 1
            self._n -= 1
            self._len_total -= len(ch.tokens)
        return removed

    def rebake_acl(self, doc_id: str, tenant: str, groups: frozenset) -> None:
        """What a re-index of one document does to the baked ACL copies."""
        for pos in self._by_doc.get(doc_id, []):
            self.chunks[pos].baked_tenant = tenant
            self.chunks[pos].baked_groups = frozenset(groups)

    @property
    def live_chunks(self) -> int:
        return self._n

    # -- searching --------------------------------------------------------
    def search(self, query: str, k: int, allow: Callable[[Chunk], bool] | None = None) -> SearchResult:
        terms = set(tokenize(query))
        avgdl = (self._len_total / self._n) if self._n else 1.0
        scores: dict = defaultdict(float)
        for t in terms:
            df = self._df.get(t, 0)
            if df <= 0:
                continue
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for pos, tf in self._post[t].items():
                ch = self.chunks[pos]
                if ch.deleted:
                    continue
                dl = len(ch.tokens)
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
                scores[pos] += idf * norm
        total = len(scores)
        denied = 0
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.chunks[kv[0]].chunk_id))
        hits = []
        for pos, sc in ranked:
            ch = self.chunks[pos]
            # Pre-filter: a forbidden chunk is skipped BEFORE it can take a slot in the top-k.
            if allow is not None and not allow(ch):
                denied += 1
                continue
            hits.append((ch, sc))
            if len(hits) >= k:
                break
        return SearchResult(hits, denied, total)
