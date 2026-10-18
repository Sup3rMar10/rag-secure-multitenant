"""A hash-chained audit log: who was shown which chunks, and proof it was not edited afterwards.

It stores identifiers and counts only. Chunk text and the question's answer never go into the log,
so the log itself is not a second copy of the restricted data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class AuditLog:
    entries: list = field(default_factory=list)
    _tick: int = 0   # a counter instead of wall-clock time keeps runs reproducible

    def record(self, *, tenant: str, user: str, groups, query: str, chunk_ids, doc_ids,
               denied: int, from_cache: bool, quarantined=(), scanner_blocked: bool = False) -> dict:
        self._tick += 1
        body = {
            "seq": self._tick, "tenant": tenant, "user": user, "groups": sorted(groups),
            "query": query, "chunk_ids": list(chunk_ids), "doc_ids": sorted(set(doc_ids)),
            "denied": denied, "from_cache": from_cache, "quarantined": list(quarantined),
            "scanner_blocked": scanner_blocked,
            "prev": self.entries[-1]["hash"] if self.entries else "genesis",
        }
        body["hash"] = self._digest(body)
        self.entries.append(body)
        return body

    @staticmethod
    def _digest(body: dict) -> str:
        payload = {k: v for k, v in body.items() if k != "hash"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def verify(self) -> bool:
        prev = "genesis"
        for e in self.entries:
            if e["prev"] != prev or e["hash"] != self._digest(e):
                return False
            prev = e["hash"]
        return True

    def who_saw(self, doc_id: str) -> list[tuple]:
        """(tenant, user) pairs whose model context contained a chunk of the document."""
        return sorted({(e["tenant"], e["user"]) for e in self.entries if doc_id in e["doc_ids"]})
