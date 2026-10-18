"""Six ways to apply permissions to retrieval. Only some of them are safe, and only some of them work."""
from __future__ import annotations

from dataclasses import dataclass, field

from .acl import AclStore, Principal, allows
from .corpus import World
from .index import Chunk

MODES = ("none", "tenant_only", "post", "post_overfetch", "pre", "partition")


@dataclass
class Retrieval:
    chunks: list
    scores: list
    denied: int = 0                 # candidates the filter rejected (audit only)
    total_matches: int = 0          # what a naive API would show as "N results"
    exposes_total: bool = False     # True when total_matches reveals documents the caller cannot read
    extra: dict = field(default_factory=dict)


def _allow_fn(principal: Principal, acl: AclStore, acl_source: str):
    if acl_source == "authoritative":
        return lambda ch: acl.can_read(principal, ch.doc_id)   # looked up NOW, so revocation is immediate
    if acl_source == "baked":
        return lambda ch: allows(ch.baked_tenant, ch.baked_groups, principal)  # a copy from index time
    raise ValueError(f"unknown acl_source {acl_source!r}")


def retrieve(world: World, principal: Principal, query: str, k: int, mode: str,
             acl_source: str = "authoritative", overfetch: int = 4) -> Retrieval:
    allow = _allow_fn(principal, world.acl, acl_source)
    if mode == "none":
        r = world.shared.search(query, k)
        return Retrieval([c for c, _ in r.hits], [s for _, s in r.hits], 0, r.total_matches)
    if mode == "tenant_only":
        # Coarse filter: keeps tenants apart, forgets that groups exist.
        r = world.shared.search(query, k, allow=lambda ch: ch.baked_tenant == principal.tenant)
        return Retrieval([c for c, _ in r.hits], [s for _, s in r.hits], r.denied, r.total_matches)
    if mode in ("post", "post_overfetch"):
        n = k if mode == "post" else k * overfetch
        r = world.shared.search(query, n)             # top-n of EVERYONE's documents...
        kept = [(c, s) for c, s in r.hits if allow(c)]   # ...then drop what the caller may not see
        kept = kept[:k]
        return Retrieval([c for c, _ in kept], [s for _, s in kept], len(r.hits) - len(kept),
                         r.total_matches, exposes_total=r.total_matches > len(kept))
    if mode == "pre":
        r = world.shared.search(query, k, allow=allow)
        return Retrieval([c for c, _ in r.hits], [s for _, s in r.hits], r.denied, len(r.hits))
    if mode == "partition":
        idx = world.partitions[principal.tenant]
        r = idx.search(query, k, allow=allow)
        return Retrieval([c for c, _ in r.hits], [s for _, s in r.hits], r.denied, len(r.hits))
    raise ValueError(f"unknown retrieval mode {mode!r}")
