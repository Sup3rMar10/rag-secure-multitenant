"""Every number in the README comes from a function here. Nothing is hard-coded."""
from __future__ import annotations

from dataclasses import replace

from .acl import Principal
from .agent import ToolPolicy, ToyAgent
from .attacks import ATTACKS, BENIGN, DEV_ATTACKS, HELDOUT_ATTACKS, by_split
from .corpus import TOPICS, build_world, workload
from .index import Chunk
from .injection import Defence, build_prompt, scan
from .pipeline import (HARDENED, NAIVE, PARTITIONED, POST_FILTER, PRE_BAKED, TENANT_ONLY, Config, SecureRAG)
from .retrieval import retrieve

K = 4


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


# ------------------------------------------------------------------ 1. retrieval
def retrieval_experiment(n_tenants: int = 3, modes=None, overfetch: int = 4, seed: int = 7) -> list[dict]:
    """Leak rate, starvation and the existence oracle of each way to apply permissions."""
    world = build_world(n_tenants, seed=seed)
    rows = []
    modes = modes or [("none", 1), ("tenant_only", 1), ("post", 1), ("post_overfetch", overfetch),
                      ("pre", 1), ("partition", 1)]
    for mode, of in modes:
        n = leaks = oracle = short = 0
        recalls, sizes = [], []
        for p, q, topic in workload(world):
            r = retrieve(world, p, q, K, mode, "authoritative", of)
            n += 1
            docs = {c.doc_id for c in r.chunks}
            leaks += any(not world.acl.can_read(p, d) for d in docs)
            oracle += r.exposes_total
            short += len(r.chunks) < min(K, len(world.relevant_docs(p, topic)) * 2)
            rel = world.relevant_docs(p, topic)
            if rel:
                recalls.append(len(docs & rel) / min(len(rel), K))
            sizes.append(len(r.chunks))
        label = mode if mode != "post_overfetch" else f"post_overfetch(x{of})"
        rows.append({"mode": label, "queries": n, "leak_pct": pct(leaks, n),
                     "mean_results": sum(sizes) / n, "recall": sum(recalls) / len(recalls),
                     "oracle_pct": pct(oracle, n), "short_pct": pct(short, n)})
    return rows


def starvation_vs_tenants(counts=(1, 3, 6, 12)) -> list[dict]:
    rows = []
    for t in counts:
        by = {r["mode"]: r for r in retrieval_experiment(
            t, modes=[("post", 1), ("post_overfetch", 4), ("post_overfetch", 16), ("pre", 1)])}
        rows.append({"tenants": t, **{m: by[m]["recall"] for m in by}})
    return rows


def idf_side_channel() -> dict:
    """Shared corpus statistics leak: another tenant adding a document changes MY scores."""
    out = {}
    for label in ("shared", "partition"):
        world = build_world(3)
        acme_dave = next(p for p in world.principals if p.tenant == "acme" and p.user == "dave")
        q = "enterprise pricing memo"

        def top_score():
            if label == "shared":
                return world.shared.search(q, 1, allow=lambda c: c.baked_tenant == "acme").hits[0][1]
            return world.partitions["acme"].search(q, 1).hits[0][1]

        before = top_score()
        from .acl import DocMeta
        # globex (a different tenant) ingests pages about pricing: acme's own text did not change.
        for i in range(30):
            world.add_doc(DocMeta(f"globex-extra-{i}", "globex", frozenset({"all"}), "extra"),
                          ["Globex enterprise pricing memo appendix. enterprise pricing memo memo memo."])
        after = top_score()
        out[label] = (before, after)
    return out


# ------------------------------------------------------------------ 2. cache
def cache_experiment() -> list[dict]:
    rows = []
    for scope in ("none", "leaky", "tenant", "scoped"):
        world = build_world(3)
        cfg = replace(HARDENED, name=f"cache={scope}", cache=scope, scanner=False)
        rag = SecureRAG(world, cfg)
        n = cross_tenant = cross_group = 0
        for p, q, _ in workload(world):
            a = rag.ask(p, q)
            n += 1
            bad = [d for d in a.doc_ids if not world.acl.can_read(p, d)]
            cross_tenant += any(world.acl.get(d).tenant != p.tenant for d in bad)
            cross_group += any(world.acl.get(d).tenant == p.tenant for d in bad)
        rows.append({"cache": scope, "queries": n, "hit_pct": pct(rag.cache.hits, n),
                     "cross_tenant_pct": pct(cross_tenant, n), "cross_group_pct": pct(cross_group, n)})
    return rows


# ------------------------------------------------------------------ 3. injection
WRAP = "Support escalation steps: page the on-call engineer within ten minutes. "
CLEAN = "Support escalation steps: page the on-call engineer within ten minutes."
FILLER = "Office hours are nine to five on working days."


def detection_experiment() -> dict:
    out = {}
    for split in ("dev", "heldout"):
        attacks, benign = by_split(ATTACKS, split), by_split(BENIGN, split)
        tp = sum(scan(WRAP + a.text).flagged for a in attacks)
        fp = sum(scan(b.text).flagged for b in benign)
        fn = len(attacks) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / len(attacks)
        out[split] = {
            "attacks": len(attacks), "benign": len(benign), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "missed": [(a.id, a.kind) for a in attacks if not scan(WRAP + a.text).flagged],
            "false_positives": [b.id for b in benign if scan(b.text).flagged],
        }
    return out


def _attack_prompt(attack, defence: Defence):
    clean = Chunk("c0", "d0", "t", CLEAN)
    bad = Chunk("c1", "d1", "t", WRAP + attack.text)
    return build_prompt("what are the support escalation steps", [clean, bad], defence)


def attack_succeeds(attack, defence: Defence, respects_delimiters: bool, tool_policy: bool) -> bool:
    res = ToyAgent(respects_delimiters).run(_attack_prompt(attack, defence))
    calls = res.tool_calls
    if tool_policy:
        calls, _ = ToolPolicy().apply(calls)
    if attack.goal_kind == "say":
        # Obeying means the whole answer was replaced. The marker merely appearing in a quoted
        # document is not a hijack (an earlier version of this check counted it as one).
        return res.obeyed_injection and res.text.strip() == attack.goal
    return ("send_email", attack.goal) in calls


def utility(defence: Defence, hard_only: bool = False) -> float:
    """Share of legitimate questions still answered correctly (a defence that blocks the user has failed too)."""
    items = [b for b in BENIGN if b.hard or not hard_only]
    ok = 0
    for b in items:
        prompt = build_prompt(b.question, [Chunk("b0", "d0", "t", b.text), Chunk("b1", "d1", "t", FILLER)], defence)
        ok += b.expect.lower() in ToyAgent().run(prompt).text.lower()
    return ok / len(items)


DEFENCES = [
    ("none", Defence()),
    ("delimit only", Defence(delimit=True)),
    ("delimit, no escaping", Defence(delimit=True, escape=False)),
    ("detect + quarantine", Defence(detect=True, action="quarantine")),
    ("detect + neutralise", Defence(detect=True, action="neutralise")),
    ("neutralise + delimit", Defence(detect=True, action="neutralise", delimit=True)),
    ("quarantine + delimit", Defence(detect=True, action="quarantine", delimit=True)),
]


def asr_experiment() -> list[dict]:
    rows = []
    for respects in (True, False):
        for name, d in DEFENCES:
            def asr(items, tool_policy=False):
                return pct(sum(attack_succeeds(a, d, respects, tool_policy) for a in items), len(items))
            dev, held = by_split(ATTACKS, "dev"), by_split(ATTACKS, "heldout")
            rows.append({"agent_respects_delimiters": respects, "defence": name,
                         "asr_dev": asr(dev), "asr_heldout": asr(held),
                         "asr_heldout_with_tool_policy": asr(held, True),
                         "utility": 100 * utility(d), "utility_hard": 100 * utility(d, True)})
    return rows


# ------------------------------------------------------------------ 4. lifecycle
def lifecycle_case(kind: str, acl_source: str, cache: str, rebake: bool = False, tombstone: bool = True) -> dict:
    """Warm the cache as a user who may read finance documents, then take that access away."""
    world = build_world(3)
    cfg = Config("lifecycle", retrieval="pre", acl_source=acl_source, cache=cache,
                 defence=Defence(), scanner=False, tool_policy=False)
    rag = SecureRAG(world, cfg)
    bobs = [p for p in world.principals if p.user == "bob"]
    topics = ("pricing", "budget")
    queries = [(b, f"what is the {TOPICS[t][0]}") for b in bobs for t in topics]
    for b, q in queries:
        rag.ask(b, q)
    for tenant in world.tenants:
        for t in topics:
            for v in (1, 2, 3):
                doc = f"{tenant}-{t}-{v}"
                if kind == "revoke":
                    world.revoke(doc, {"exec"}, rebake=rebake)
                else:
                    world.delete(doc, tombstone=tombstone)
    leaks = 0
    for b, q in queries:
        a = rag.ask(b, q)
        leaks += any(not world.acl.can_read(b, d) for d in a.doc_ids)
    return {"kind": kind, "acl_source": acl_source, "cache": cache, "rebake": rebake,
            "tombstone": tombstone, "queries": len(queries), "leaks": leaks, "leak_pct": pct(leaks, len(queries))}


def lifecycle_experiment() -> list[dict]:
    cases = [
        ("revoke", "baked", "none", {"rebake": False}),
        ("revoke", "baked", "none", {"rebake": True}),
        ("revoke", "authoritative", "none", {"rebake": False}),
        ("revoke", "authoritative", "scoped_noepoch", {"rebake": False}),
        ("revoke", "authoritative", "scoped", {"rebake": False}),
        ("delete", "baked", "none", {"tombstone": False}),
        ("delete", "baked", "none", {"tombstone": True}),
        ("delete", "authoritative", "none", {"tombstone": False}),
        ("delete", "authoritative", "scoped_noepoch", {"tombstone": True}),
        ("delete", "authoritative", "scoped", {"tombstone": True}),
    ]
    return [lifecycle_case(k, a, c, **kw) for k, a, c, kw in cases]


# ------------------------------------------------------------------ 5. end to end
HARDENED_UNTRUSTED_DELIMS = replace(HARDENED, name="hardened (agent ignores delimiters)", respects_delimiters=False)
VARIANTS = [NAIVE, TENANT_ONLY, POST_FILTER, PRE_BAKED, HARDENED, PARTITIONED, HARDENED_UNTRUSTED_DELIMS]


# One override, one exfiltration, one forged role marker: the common, text-only attacks.
MAIN_PAYLOADS = [a.text for a in DEV_ATTACKS if a.id in ("D01", "D06", "D03")]


def matrix_experiment(poison_payloads=None, variants=None) -> list[dict]:
    poison_payloads = poison_payloads or MAIN_PAYLOADS
    rows = []
    for cfg in variants or VARIANTS:
        world = build_world(3, poison=True, poison_payloads=poison_payloads)
        rag = SecureRAG(world, cfg)
        n = ctx = ans = cross = blocked = 0
        exposed = hijacked = exfil = 0   # exposed: the poisoned chunk reached the model's context
        recalls, sizes = [], []
        for p, q, topic in workload(world):
            a = rag.ask(p, q)
            n += 1
            bad = [d for d in a.doc_ids if not world.acl.can_read(p, d)]
            ctx += bool(bad)
            cross += any(world.acl.get(d).tenant != p.tenant for d in bad)
            ans += bool(world.canaries.leaks(a.text, p, world.acl))
            blocked += a.scanner_blocked
            poison_ids = {f"{d}#1" for d in world.poison_docs}
            if poison_ids & set(a.chunk_ids + a.quarantined):
                exposed += 1
                hijacked += a.obeyed_injection and not a.tool_calls
                exfil += bool(a.tool_calls)
            rel = world.relevant_docs(p, topic)
            if rel:
                recalls.append(len(set(a.doc_ids) & rel) / min(len(rel), K))
            sizes.append(len(a.chunk_ids))
        rows.append({"variant": cfg.name, "queries": n, "ctx_leak_pct": pct(ctx, n),
                     "cross_tenant_pct": pct(cross, n), "answer_leak_pct": pct(ans, n),
                     "scanner_blocked": blocked, "recall": sum(recalls) / len(recalls),
                     "mean_results": sum(sizes) / n, "exposed_queries": exposed,
                     "hijack_pct": pct(hijacked, exposed), "exfil_pct": pct(exfil, exposed)})
    return rows


def scanner_coverage(rates=(0.0, 0.25, 0.5, 1.0)) -> list[dict]:
    """The canary scanner only knows documents that carry a canary. How much of a real leak does it catch?"""
    rows = []
    for rate in rates:
        world = build_world(3, canary_rate=rate)
        cfg = replace(TENANT_ONLY, scanner=True, cache="none")
        rag = SecureRAG(world, cfg)
        leaks = caught = 0
        for p, q, _ in workload(world):
            a = rag.ask(p, q)
            truly = any(not world.acl.can_read(p, d) for d in a.doc_ids)
            # an actual leak is any restricted chunk in the context; an answer leak is what the user could read.
            leaks += truly
            caught += truly and a.scanner_blocked
        rows.append({"canary_rate": rate, "leaky_queries": leaks, "caught": caught, "caught_pct": pct(caught, leaks)})
    return rows


def audit_blast_radius() -> dict:
    """After a leak is discovered, who saw the document? Run on the naive variant, then verify the chain."""
    world = build_world(3)
    rag = SecureRAG(world, replace(NAIVE, cache="none"))
    for p, q, _ in workload(world):
        rag.ask(p, q)
    doc = "acme-pricing-2"
    seen = rag.audit.who_saw(doc)
    entitled = [(t, u) for t, u in seen
                if world.acl.can_read(next(p for p in world.principals if (p.tenant, p.user) == (t, u)), doc)]
    return {"doc": doc, "viewers": len(seen), "entitled": len(entitled), "not_entitled": len(seen) - len(entitled),
            "chain_ok": rag.audit.verify(), "entries": len(rag.audit.entries)}
