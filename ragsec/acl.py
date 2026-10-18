"""The authoritative permission model.

A document belongs to exactly one tenant and lists the groups allowed to read it.
The group ``all`` means "every member of the owning tenant". Nothing ever crosses
a tenant boundary, whatever the groups say.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

ALL = "all"


@dataclass(frozen=True)
class Principal:
    tenant: str
    user: str
    groups: frozenset = frozenset()

    def fingerprint(self) -> str:
        """Stable id of (tenant, groups). Two users with the same fingerprint may share cache entries."""
        raw = self.tenant + "|" + ",".join(sorted(self.groups))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class DocMeta:
    doc_id: str
    tenant: str
    groups: frozenset
    topic: str = ""
    version: int = 1
    deleted: bool = False

    @property
    def restricted(self) -> bool:
        return ALL not in self.groups


def allows(tenant: str, groups: frozenset, principal: Principal) -> bool:
    """The one rule, shared by the authoritative store and by 'baked' chunk copies."""
    if tenant != principal.tenant:
        return False
    return ALL in groups or bool(groups & principal.groups)


@dataclass
class AclStore:
    docs: dict = field(default_factory=dict)
    _epoch: dict = field(default_factory=dict)

    def add(self, meta: DocMeta) -> None:
        self.docs[meta.doc_id] = meta
        self._bump(meta.tenant)

    def get(self, doc_id: str) -> DocMeta | None:
        return self.docs.get(doc_id)

    def can_read(self, principal: Principal, doc_id: str) -> bool:
        meta = self.docs.get(doc_id)
        if meta is None or meta.deleted:
            return False
        return allows(meta.tenant, meta.groups, principal)

    def set_groups(self, doc_id: str, groups) -> None:
        meta = self.docs[doc_id]
        meta.groups = frozenset(groups)
        meta.version += 1
        self._bump(meta.tenant)

    def delete(self, doc_id: str) -> None:
        meta = self.docs[doc_id]
        meta.deleted = True
        meta.version += 1
        self._bump(meta.tenant)

    def epoch(self, tenant: str) -> int:
        """Bumped on every permission or deletion change in the tenant; part of the scoped cache key."""
        return self._epoch.get(tenant, 0)

    def _bump(self, tenant: str) -> None:
        self._epoch[tenant] = self._epoch.get(tenant, 0) + 1
