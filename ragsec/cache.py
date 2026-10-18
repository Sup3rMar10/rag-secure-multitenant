"""A semantic answer cache. Whether it is safe depends entirely on what goes into the key."""
from __future__ import annotations

from dataclasses import dataclass, field

from .acl import AclStore, Principal
from .text import jaccard, tokenize

SCOPES = ("none", "leaky", "tenant", "scoped_noepoch", "scoped")


@dataclass
class SemanticCache:
    scope: str = "scoped"
    threshold: float = 0.8
    entries: dict = field(default_factory=dict)   # (scope key, frozenset(tokens)) -> value
    hits: int = 0
    misses: int = 0

    def scope_key(self, principal: Principal, acl: AclStore) -> str:
        if self.scope == "leaky":
            return ""                                            # same question, same answer, for anyone
        if self.scope == "tenant":
            return principal.tenant                              # separates tenants, not groups
        if self.scope == "scoped_noepoch":
            return f"{principal.tenant}|{principal.fingerprint()}"   # right asker, but blind to ACL changes
        if self.scope == "scoped":
            return f"{principal.tenant}|{principal.fingerprint()}|{acl.epoch(principal.tenant)}"
        raise ValueError(f"cache scope {self.scope!r} is not keyed")

    def get(self, principal: Principal, query: str, acl: AclStore):
        if self.scope == "none":
            return None
        sk = self.scope_key(principal, acl)
        bag = frozenset(tokenize(query))
        for (k2, bag2), value in self.entries.items():
            if k2 == sk and jaccard(bag, bag2) >= self.threshold:
                self.hits += 1
                return value
        self.misses += 1
        return None

    def put(self, principal: Principal, query: str, acl: AclStore, value) -> None:
        if self.scope == "none":
            return
        self.entries[(self.scope_key(principal, acl), frozenset(tokenize(query)))] = value
