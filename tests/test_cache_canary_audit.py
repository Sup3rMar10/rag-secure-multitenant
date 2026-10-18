import copy

import pytest

from ragsec.acl import Principal
from ragsec.audit import AuditLog
from ragsec.cache import SemanticCache
from ragsec.canary import CanaryRegistry, make_canary
from ragsec.corpus import build_world
from ragsec.text import tokenize


def P(tenant, groups=()):
    return Principal(tenant, "u", frozenset(groups))


@pytest.fixture()
def acl():
    return build_world(2).acl


def test_leaky_key_serves_one_tenants_answer_to_another(acl):
    c = SemanticCache("leaky")
    c.put(P("acme", ["finance"]), "enterprise pricing memo", acl, "ACME SECRET")
    assert c.get(P("globex"), "enterprise pricing memo", acl) == "ACME SECRET"


def test_tenant_key_stops_tenant_leaks_but_not_group_leaks(acl):
    c = SemanticCache("tenant")
    c.put(P("acme", ["finance"]), "enterprise pricing memo", acl, "FINANCE ANSWER")
    assert c.get(P("globex"), "enterprise pricing memo", acl) is None
    assert c.get(P("acme"), "enterprise pricing memo", acl) == "FINANCE ANSWER"   # the bug


def test_scoped_key_isolates_tenant_and_groups(acl):
    c = SemanticCache("scoped")
    c.put(P("acme", ["finance"]), "enterprise pricing memo", acl, "A")
    assert c.get(P("acme"), "enterprise pricing memo", acl) is None
    assert c.get(P("globex", ["finance"]), "enterprise pricing memo", acl) is None
    assert c.get(P("acme", ["finance"]), "enterprise pricing memo", acl) == "A"


def test_scoped_key_shares_between_users_with_the_same_groups(acl):
    c = SemanticCache("scoped")
    c.put(Principal("acme", "bob", frozenset({"finance"})), "pricing memo", acl, "A")
    assert c.get(Principal("acme", "erin", frozenset({"finance"})), "pricing memo", acl) == "A"


def test_epoch_invalidates_after_an_acl_change_and_noepoch_does_not(acl):
    p = P("acme", ["finance"])
    scoped, stale = SemanticCache("scoped"), SemanticCache("scoped_noepoch")
    for c in (scoped, stale):
        c.put(p, "quarterly budget forecast", acl, "OLD")
    acl.set_groups("acme-budget-1", {"exec"})
    assert scoped.get(p, "quarterly budget forecast", acl) is None
    assert stale.get(p, "quarterly budget forecast", acl) == "OLD"


def test_semantic_match_is_by_token_overlap(acl):
    c = SemanticCache("scoped", threshold=0.8)
    p = P("acme")
    c.put(p, "what is the enterprise pricing memo", acl, "A")
    assert c.get(p, "memo pricing enterprise", acl) == "A"          # same bag of words
    assert c.get(p, "enterprise roadmap memo", acl) is None         # different question


def test_none_scope_never_stores(acl):
    c = SemanticCache("none")
    c.put(P("acme"), "q words", acl, "A")
    assert c.get(P("acme"), "q words", acl) is None and not c.entries


def test_unknown_scope_is_an_error(acl):
    with pytest.raises(ValueError):
        SemanticCache("bogus").get(P("acme"), "q", acl)


# ---- canaries
def test_canary_is_stable_and_unique_per_document():
    assert make_canary("s", "d1") == make_canary("s", "d1")
    assert make_canary("s", "d1") != make_canary("s", "d2")
    assert make_canary("s", "d1") != make_canary("other", "d1")


@pytest.mark.parametrize("wrap", [
    "{c}", "see {c}.", "{cl}", "C A N A R Y - {h}", "CANARY​-{h}", "canary {h}", "Canary-{hu}",
    "ＣＡＮＡＲＹ-{h}",     # full-width letters need NFKC
    "CANаRY-{h}",                               # Cyrillic a needs the homoglyph map
])
def test_scanner_finds_obfuscated_canaries(wrap):
    reg = CanaryRegistry()
    c = make_canary("s", "doc")
    reg.register(c, "doc")
    h = c.split("-")[1]
    assert reg.find(wrap.format(c=c, cl=c.lower(), h=h, hu=h.upper())) == ["doc"]


def test_scanner_has_no_false_positive_on_ordinary_text():
    reg = CanaryRegistry()
    reg.register(make_canary("s", "doc"), "doc")
    assert reg.find("The enterprise pricing memo covers seat tiers and renewals.") == []


def test_scanner_misses_a_base64_encoded_canary_and_that_is_documented():
    import base64
    reg = CanaryRegistry()
    c = make_canary("s", "doc")
    reg.register(c, "doc")
    assert reg.find(base64.b64encode(c.encode()).decode()) == []


def test_leaks_reports_only_documents_the_asker_cannot_read():
    w = build_world(2)
    reg = w.canaries
    doc = next(iter(reg.by_canary.values()))
    meta = w.acl.get(doc)
    text = f"quoting {make_canary(w.secret, doc)}"
    owner = next(p for p in w.principals if p.tenant == meta.tenant and w.acl.can_read(p, doc))
    outsider = next(p for p in w.principals if p.tenant != meta.tenant)
    assert reg.leaks(text, owner, w.acl) == []
    assert reg.leaks(text, outsider, w.acl) == [doc]


def test_planting_a_canary_does_not_change_tokens():
    c = make_canary("s", "doc")
    assert tokenize(f"pricing memo, ref {c}, version 1") == tokenize("pricing memo, ref , version 1")


# ---- audit
def _log(n=3):
    log = AuditLog()
    for i in range(n):
        log.record(tenant="acme", user=f"u{i}", groups=["finance"], query="q", chunk_ids=[f"d{i}#0"],
                   doc_ids=[f"d{i}"], denied=i, from_cache=False)
    return log


def test_chain_verifies_when_untouched():
    assert _log().verify()


def test_editing_an_entry_breaks_the_chain():
    log = _log()
    log.entries[1]["user"] = "someone-else"
    assert not log.verify()


def test_deleting_an_entry_breaks_the_chain():
    log = _log()
    del log.entries[1]
    assert not log.verify()


def test_reordering_entries_breaks_the_chain():
    log = _log()
    log.entries[0], log.entries[1] = log.entries[1], log.entries[0]
    assert not log.verify()


def test_who_saw_names_every_viewer_of_a_document():
    log = _log()
    log.record(tenant="globex", user="x", groups=[], query="q", chunk_ids=["d0#1"], doc_ids=["d0"],
               denied=0, from_cache=True)
    assert log.who_saw("d0") == [("acme", "u0"), ("globex", "x")]
    assert log.who_saw("missing") == []


def test_log_holds_identifiers_not_text():
    entry = _log(1).entries[0]
    assert set(entry) >= {"chunk_ids", "doc_ids", "denied"}
    assert "text" not in entry and "answer" not in entry


def test_log_is_reproducible():
    assert [e["hash"] for e in _log().entries] == [e["hash"] for e in copy.deepcopy(_log()).entries]
