"""The whole path: cache -> permission-aware retrieval -> injection defence -> agent -> tool policy -> output scan -> audit."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .acl import Principal
from .agent import ToolPolicy, ToyAgent
from .audit import AuditLog
from .cache import SemanticCache
from .corpus import World
from .injection import Defence, build_prompt
from .retrieval import retrieve

BLOCKED_TEXT = "[blocked: the answer contained restricted content]"


@dataclass(frozen=True)
class Config:
    name: str = "hardened"
    retrieval: str = "pre"
    acl_source: str = "authoritative"
    cache: str = "scoped"
    defence: Defence = Defence(detect=True, action="neutralise", delimit=True)
    scanner: bool = True
    tool_policy: bool = True
    respects_delimiters: bool = True
    k: int = 4
    overfetch: int = 4


@dataclass
class Answer:
    text: str
    chunk_ids: list
    doc_ids: list
    tool_calls: list = field(default_factory=list)      # executed
    blocked_calls: list = field(default_factory=list)
    from_cache: bool = False
    scanner_blocked: bool = False
    quarantined: list = field(default_factory=list)
    obeyed_injection: bool = False
    exposes_total: bool = False


class SecureRAG:
    def __init__(self, world: World, config: Config):
        self.world, self.config = world, config
        self.cache = SemanticCache(config.cache)
        self.audit = AuditLog()
        self.agent = ToyAgent(config.respects_delimiters)
        self.policy = ToolPolicy(frozenset() if config.tool_policy else frozenset({"send_email"}))

    def ask(self, principal: Principal, query: str) -> Answer:
        w, c = self.world, self.config
        cached = self.cache.get(principal, query, w.acl)
        if cached is not None:
            ans = replace(cached, from_cache=True)
            denied = 0
        else:
            r = retrieve(w, principal, query, c.k, c.retrieval, c.acl_source, c.overfetch)
            prompt = build_prompt(query, r.chunks, c.defence)
            res = self.agent.run(prompt)
            executed, blocked = self.policy.apply(res.tool_calls)
            kept_ids = {cid for cid, _ in prompt.docs}
            chunk_ids = [ch.chunk_id for ch in r.chunks if ch.chunk_id in kept_ids]
            ans = Answer(res.text, chunk_ids, [i.split("#")[0] for i in chunk_ids], executed, blocked,
                         quarantined=prompt.quarantined, obeyed_injection=res.obeyed_injection,
                         exposes_total=r.exposes_total)
            denied = r.denied
            self.cache.put(principal, query, w.acl, ans)
        if c.scanner and w.canaries.leaks(ans.text, principal, w.acl):
            ans = replace(ans, text=BLOCKED_TEXT, scanner_blocked=True)
        self.audit.record(tenant=principal.tenant, user=principal.user, groups=principal.groups,
                          query=query, chunk_ids=ans.chunk_ids, doc_ids=ans.doc_ids, denied=denied,
                          from_cache=ans.from_cache, quarantined=ans.quarantined,
                          scanner_blocked=ans.scanner_blocked)
        return ans


# Named variants used by the experiments and the README.
NAIVE = Config("naive", retrieval="none", cache="leaky", defence=Defence(), scanner=False, tool_policy=False,
               respects_delimiters=False)
TENANT_ONLY = replace(NAIVE, name="tenant-filter-only", retrieval="tenant_only", cache="tenant")
POST_FILTER = replace(NAIVE, name="post-filter", retrieval="post", cache="scoped")
PRE_BAKED = replace(NAIVE, name="pre-filter+baked-acl", retrieval="pre", acl_source="baked",
                    cache="scoped_noepoch")
HARDENED = Config("hardened")
PARTITIONED = replace(HARDENED, name="hardened+partition", retrieval="partition")
