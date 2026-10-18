"""Canary strings: unique tokens planted in restricted documents, hunted for in every output."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from .acl import AclStore, Principal
from .normalise import flat


def make_canary(secret: str, doc_id: str) -> str:
    digest = hmac.new(secret.encode(), doc_id.encode(), hashlib.sha256).hexdigest()[:10]
    return f"CANARY-{digest}"


@dataclass
class CanaryRegistry:
    by_canary: dict = field(default_factory=dict)  # canary -> doc_id

    def register(self, canary: str, doc_id: str) -> None:
        self.by_canary[canary] = doc_id

    def find(self, text: str) -> list[str]:
        """Doc ids whose canary appears in text, tolerating spacing, punctuation, case and zero-width tricks."""
        hay = flat(text)
        return sorted(doc for canary, doc in self.by_canary.items() if flat(canary) in hay)

    def leaks(self, text: str, principal: Principal, acl: AclStore) -> list[str]:
        return [d for d in self.find(text) if not acl.can_read(principal, d)]
