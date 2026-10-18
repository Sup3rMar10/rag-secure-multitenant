from ragsec.acl import AclStore, DocMeta, Principal, allows
from ragsec.corpus import ROLE_GROUPS, TOPICS, build_world, workload


def P(tenant="acme", user="u", groups=()):
    return Principal(tenant, user, frozenset(groups))


def test_all_group_means_whole_tenant_only():
    assert allows("acme", frozenset({"all"}), P("acme"))
    assert not allows("acme", frozenset({"all"}), P("globex"))


def test_group_must_match():
    assert allows("acme", frozenset({"finance"}), P("acme", groups=["finance"]))
    assert not allows("acme", frozenset({"finance"}), P("acme", groups=["eng"]))
    assert not allows("acme", frozenset({"finance"}), P("acme"))


def test_matching_group_never_crosses_a_tenant_boundary():
    assert not allows("acme", frozenset({"finance"}), P("globex", groups=["finance"]))


def test_deleted_document_is_unreadable_even_for_its_owner():
    acl = AclStore()
    acl.add(DocMeta("d", "acme", frozenset({"all"})))
    acl.delete("d")
    assert not acl.can_read(P("acme"), "d")


def test_unknown_document_is_unreadable():
    assert not AclStore().can_read(P(), "nope")


def test_epoch_bumps_on_every_permission_change():
    acl = AclStore()
    acl.add(DocMeta("d", "acme", frozenset({"all"})))
    e0 = acl.epoch("acme")
    acl.set_groups("d", {"eng"})
    e1 = acl.epoch("acme")
    acl.delete("d")
    assert e0 < e1 < acl.epoch("acme")
    assert acl.epoch("globex") == 0


def test_fingerprint_is_order_independent_and_tenant_sensitive():
    assert P("acme", "a", ["x", "y"]).fingerprint() == P("acme", "b", ["y", "x"]).fingerprint()
    assert P("acme", "a", ["x"]).fingerprint() != P("globex", "a", ["x"]).fingerprint()
    assert P("acme", "a", ["x"]).fingerprint() != P("acme", "a", ["x", "y"]).fingerprint()


def test_world_is_deterministic():
    a, b = build_world(3, seed=5), build_world(3, seed=5)
    assert [c.text for c in a.shared.chunks] == [c.text for c in b.shared.chunks]
    assert a.canaries.by_canary == b.canaries.by_canary


def test_every_tenant_has_every_topic_in_the_same_words():
    w = build_world(3)
    for t in w.tenants:
        topics = {m.topic for m in w.acl.docs.values() if m.tenant == t}
        assert topics == set(TOPICS)


def test_restricted_docs_get_canaries_and_public_docs_do_not():
    w = build_world(3)
    registered = set(w.canaries.by_canary.values())
    for d, m in w.acl.docs.items():
        assert (d in registered) == m.restricted


def test_canary_rate_zero_plants_nothing():
    assert not build_world(3, canary_rate=0.0).canaries.by_canary


def test_poison_docs_are_public_and_in_every_tenant():
    w = build_world(3, poison=True)
    assert len(w.poison_docs) == 3
    for d in w.poison_docs:
        assert not w.acl.get(d).restricted


def test_every_chunk_repeats_its_document_header():
    w = build_world(2)
    for c in w.shared.chunks:
        header = c.text.split(" (part")[0]
        assert header in w.texts[c.doc_id][0] and header in w.texts[c.doc_id][1]


def test_workload_asks_privileged_users_first():
    w = build_world(3)
    users = [p.user for p, _, _ in workload(w)]
    order = [u for u, _ in ROLE_GROUPS]
    assert users == sorted(users, key=order.index)


def test_relevant_docs_respect_acl():
    w = build_world(3)
    dave = next(p for p in w.principals if p.user == "dave" and p.tenant == "acme")
    carol = next(p for p in w.principals if p.user == "carol" and p.tenant == "acme")
    assert w.relevant_docs(dave, "pricing") == set()
    assert len(w.relevant_docs(carol, "pricing")) == 3
