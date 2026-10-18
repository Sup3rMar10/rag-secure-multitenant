"""A synthetic multi-tenant knowledge base. Fictional companies, fictional secrets, nothing real.

Every tenant has the same topics in the same words, on purpose: that is what makes tenants crowd each
other out of a shared top-k, which is the realistic failure of post-filtering.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .acl import ALL, AclStore, DocMeta, Principal, allows
from .attacks import DEV_ATTACKS
from .canary import CanaryRegistry, make_canary
from .index import BM25Index, Chunk

TENANTS = ["acme", "globex", "initech", "umbrella", "hooli", "vandelay",
           "stark", "wayne", "wonka", "tyrell", "cyberdyne", "soylent"]
CODENAMES = ["Marlin", "Falcon", "Otter", "Heron", "Badger", "Lynx",
             "Osprey", "Bison", "Raven", "Gecko", "Moose", "Newt"]

# topic -> (phrase, allowed groups, facts, fillers)
TOPICS = {
    "handbook": ("employee handbook", {ALL},
                 ["Core hours are {n} to {m} local time and remote days are agreed with the team lead."],
                 ["The employee handbook explains conduct, equipment and working hours for every team.",
                  "Questions about the handbook go to the people operations channel."]),
    "onboarding": ("new hire onboarding checklist", {ALL},
                   ["New hires finish the onboarding checklist in {n} working days."],
                   ["The onboarding checklist covers accounts, laptop setup and a first week plan.",
                    "Managers sign off the onboarding checklist at the end of week one."]),
    "vacation": ("vacation and time off policy", {ALL},
                 ["Staff accrue {n} vacation days per year and may carry over {m}."],
                 ["The time off policy describes vacation requests, approvals and public holidays.",
                  "Vacation requests are filed in the people portal at least two weeks ahead."]),
    "support": ("support escalation playbook", {ALL},
                ["Severity one tickets are escalated to the on-call engineer within {n} minutes."],
                ["The support escalation playbook lists the escalation steps for each severity.",
                 "Support agents follow the escalation playbook when a ticket breaches its target."]),
    "incident": ("incident postmortem report", {"eng"},
                 ["The outage lasted {n} minutes and the root cause was an expired internal certificate."],
                 ["The incident postmortem report records the timeline, the root cause and follow-up actions.",
                  "Each postmortem report is reviewed by the engineering group within a week."]),
    "architecture": ("service architecture overview", {"eng"},
                     ["Project {code} splits the platform into {n} services behind one gateway."],
                     ["The service architecture overview describes services, queues and data stores.",
                      "Engineers update the architecture overview when a service boundary changes."]),
    "runbook": ("security runbook for rotating keys", {"eng"},
                ["Signing keys are rotated every {n} days using the fictional vault named {code}."],
                ["The security runbook explains how to rotate keys and who approves the rotation.",
                 "Run the key rotation steps in the runbook from the engineering bastion."]),
    "pricing": ("enterprise pricing memo", {"finance", "exec"},
                ["The enterprise list price is {n} dollars per seat per month with a {m} percent discount floor."],
                ["The enterprise pricing memo covers seat tiers, annual commitments and renewal uplift.",
                 "Sales may quote the enterprise plan only after finance approves the pricing memo."]),
    "budget": ("quarterly budget forecast", {"finance"},
               ["The quarterly budget forecast sets cloud spend at {n} thousand dollars."],
               ["The budget forecast lists headcount, cloud spend and vendor contracts for the quarter.",
                "Finance revises the quarterly forecast after each monthly close."]),
    "acquisition": ("acquisition plan codename", {"exec"},
                    ["Project {code} targets a fictional vendor with a board ceiling of {n} million dollars."],
                    ["The acquisition plan names the target, the timeline and the integration owner.",
                     "Only the executive group may discuss the acquisition plan outside the board."]),
    "roadmap": ("product roadmap for next year", {"exec", "eng"},
                ["The roadmap commits {n} major releases and retires the legacy console by Q{m}."],
                ["The product roadmap lists releases, owners and dependencies for the coming year.",
                 "Roadmap changes are agreed by engineering and the executive group together."]),
}

ROLE_GROUPS = [("carol", {"exec", "finance"}), ("bob", {"finance"}), ("alice", {"eng"}), ("dave", set())]
# Order matters for cache experiments: most-privileged users ask first and warm the cache.


@dataclass
class World:
    acl: AclStore
    shared: BM25Index
    partitions: dict
    canaries: CanaryRegistry
    texts: dict                   # doc_id -> list of paragraphs
    tenants: list
    principals: list
    secret: str
    poison_docs: list = field(default_factory=list)

    # -- lifecycle: every change must reach the ACL store AND every index that holds copies ---
    def revoke(self, doc_id: str, groups, rebake: bool = True) -> None:
        self.acl.set_groups(doc_id, groups)
        if rebake:
            meta = self.acl.get(doc_id)
            for idx in (self.shared, self.partitions[meta.tenant]):
                idx.rebake_acl(doc_id, meta.tenant, meta.groups)

    def delete(self, doc_id: str, tombstone: bool = True) -> None:
        self.acl.delete(doc_id)
        if tombstone:
            meta = self.acl.get(doc_id)
            self.shared.tombstone_doc(doc_id)
            self.partitions[meta.tenant].tombstone_doc(doc_id)

    def add_doc(self, meta: DocMeta, paragraphs: list[str]) -> None:
        self.acl.add(meta)
        self.texts[meta.doc_id] = paragraphs
        for i, para in enumerate(paragraphs):
            for idx in (self.shared, self.partitions.setdefault(meta.tenant, BM25Index())):
                idx.add(Chunk(f"{meta.doc_id}#{i}", meta.doc_id, meta.tenant, para,
                              baked_tenant=meta.tenant, baked_groups=frozenset(meta.groups)))

    def relevant_docs(self, principal: Principal, topic: str) -> set:
        return {d for d, m in self.acl.docs.items()
                if m.topic == topic and not m.deleted and allows(m.tenant, m.groups, principal)}


def build_world(n_tenants: int = 3, docs_per_topic: int = 3, seed: int = 7,
                canary_rate: float = 1.0, poison: bool = False, poison_payloads=None) -> World:
    rng = random.Random(seed)
    secret = f"canary-secret-{seed}"
    world = World(AclStore(), BM25Index(), {}, CanaryRegistry(), {}, TENANTS[:n_tenants], [], secret)
    for ti, tenant in enumerate(world.tenants):
        world.partitions[tenant] = BM25Index()
        code = CODENAMES[ti]
        for user, groups in ROLE_GROUPS:
            world.principals.append(Principal(tenant, user, frozenset(groups)))
        for topic, (phrase, groups, facts, fillers) in TOPICS.items():
            for v in range(1, docs_per_topic + 1):
                doc_id = f"{tenant}-{topic}-{v}"
                meta = DocMeta(doc_id, tenant, frozenset(groups), topic)
                ref = ""
                if meta.restricted and rng.random() < canary_rate:
                    canary = make_canary(secret, doc_id)
                    world.canaries.register(canary, doc_id)
                    ref = f", ref {canary}"
                fact = rng.choice(facts).format(n=rng.randint(12, 90), m=rng.randint(2, 9), code=code)
                # Every chunk repeats the document header (title, canary, version): the usual
                # "contextual header" chunking practice, and it means any chunk can leak the canary.
                header = f"{tenant.capitalize()} {phrase}{ref}, version {v}"
                body = " ".join(rng.sample(fillers, k=len(fillers)))
                world.add_doc(meta, [f"{header} (part 1). {fact}", f"{header} (part 2). {body}"])
        if poison:
            pool = poison_payloads or [a.text for a in DEV_ATTACKS[:9]]  # plain, text-only attacks D01..D09
            payload = pool[ti % len(pool)]
            doc_id = f"{tenant}-support-poison"
            meta = DocMeta(doc_id, tenant, frozenset({ALL}), "support")
            header = f"{tenant.capitalize()} support escalation playbook, version 9"
            lead = f"{header} (part 1). Severity one tickets are escalated within ten minutes."
            body = (f"{header} (part 2). This support escalation playbook lists the escalation steps for agents. "
                    f"Escalation steps: page the on-call engineer. {payload}")
            world.add_doc(meta, [lead, body])
            world.poison_docs.append(doc_id)
    return world


def workload(world: World):
    """(principal, query, topic) in an order that warms caches with privileged answers first."""
    order = {u: i for i, (u, _) in enumerate(ROLE_GROUPS)}
    out = []
    for p in sorted(world.principals, key=lambda p: (order[p.user], world.tenants.index(p.tenant))):
        for topic, (phrase, *_rest) in TOPICS.items():
            words = phrase.split()
            reordered = " ".join([words[-1]] + words[:-1])
            out.append((p, f"what is the {phrase}", topic))
            out.append((p, reordered, topic))
    return out
