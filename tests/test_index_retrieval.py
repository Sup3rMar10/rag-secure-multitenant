import pytest

from ragsec.corpus import build_world
from ragsec.index import BM25Index, Chunk
from ragsec.retrieval import MODES, retrieve


def mk(cid, doc, text, tenant="t"):
    return Chunk(cid, doc, tenant, text, baked_tenant=tenant, baked_groups=frozenset({"all"}))


def test_bm25_ranks_the_matching_chunk_first():
    idx = BM25Index()
    idx.add(mk("a", "a", "enterprise pricing memo for seats"))
    idx.add(mk("b", "b", "holiday calendar for the office"))
    assert idx.search("pricing memo", 1).hits[0][0].chunk_id == "a"


def test_prefilter_fills_topk_with_allowed_chunks():
    idx = BM25Index()
    for i in range(10):
        idx.add(mk(f"x{i}", f"x{i}", "secret pricing memo pricing"))      # strong match, forbidden
    for i in range(3):
        idx.add(mk(f"ok{i}", f"ok{i}", "pricing notes"))                   # weaker match, allowed
    res = idx.search("pricing memo", 3, allow=lambda c: c.chunk_id.startswith("ok"))
    assert [c.chunk_id for c, _ in res.hits] == ["ok0", "ok1", "ok2"]
    assert res.denied == 10


def test_total_matches_counts_before_filtering():
    idx = BM25Index()
    for i in range(5):
        idx.add(mk(f"x{i}", f"x{i}", "pricing"))
    assert idx.search("pricing", 1, allow=lambda c: False).total_matches == 5


def test_tombstone_removes_chunk_and_restores_exact_statistics():
    docs = [("a", "pricing memo"), ("b", "pricing policy"), ("c", "holiday policy")]
    full, ref = BM25Index(), BM25Index()
    for d, t in docs:
        full.add(mk(d, d, t))
    for d, t in docs[:2]:
        ref.add(mk(d, d, t))
    assert full.tombstone_doc("c") == 1
    q = "pricing policy"
    a = [(c.chunk_id, round(s, 9)) for c, s in full.search(q, 5).hits]
    b = [(c.chunk_id, round(s, 9)) for c, s in ref.search(q, 5).hits]
    assert a == b   # scores equal an index that never saw the deleted document


def test_tombstone_twice_is_harmless():
    idx = BM25Index()
    idx.add(mk("a", "a", "x y"))
    assert idx.tombstone_doc("a") == 1
    assert idx.tombstone_doc("a") == 0
    assert idx.live_chunks == 0


def test_search_is_deterministic_on_ties():
    idx = BM25Index()
    for c in ("b", "a", "c"):
        idx.add(mk(c, c, "same words"))
    assert [c.chunk_id for c, _ in idx.search("same words", 3).hits] == ["a", "b", "c"]


@pytest.fixture(scope="module")
def w():
    return build_world(6)


def _user(w, tenant, user):
    return next(p for p in w.principals if (p.tenant, p.user) == (tenant, user))


def _leaks(w, p, r):
    return [c.doc_id for c in r.chunks if not w.acl.can_read(p, c.doc_id)]


def test_unfiltered_retrieval_leaks_other_tenants(w):
    r = retrieve(w, _user(w, "acme", "carol"), "enterprise pricing memo", 4, "none")
    assert _leaks(w, _user(w, "acme", "carol"), r)


def test_tenant_only_filter_keeps_tenants_apart_but_leaks_groups(w):
    dave = _user(w, "acme", "dave")
    r = retrieve(w, dave, "enterprise pricing memo", 4, "tenant_only")
    assert all(c.tenant == "acme" for c in r.chunks)
    assert _leaks(w, dave, r)          # dave is in no group, but sees finance documents


@pytest.mark.parametrize("mode", ["post", "post_overfetch", "pre", "partition"])
def test_correct_modes_never_leak(w, mode):
    for p in w.principals:
        r = retrieve(w, p, "enterprise pricing memo", 4, mode)
        assert not _leaks(w, p, r)


def test_post_filter_starves_the_topk(w):
    carol = _user(w, "acme", "carol")
    post = retrieve(w, carol, "enterprise pricing memo", 4, "post")
    pre = retrieve(w, carol, "enterprise pricing memo", 4, "pre")
    assert len(pre.chunks) == 4
    assert len(post.chunks) < len(pre.chunks)


def test_post_filter_exposes_an_existence_oracle_and_prefilter_does_not(w):
    dave = _user(w, "acme", "dave")
    post = retrieve(w, dave, "acquisition plan codename", 4, "post")
    pre = retrieve(w, dave, "acquisition plan codename", 4, "pre")
    assert post.exposes_total and post.total_matches > len(post.chunks)
    assert not pre.exposes_total and pre.total_matches == len(pre.chunks)


def test_overfetch_repairs_starvation_only_up_to_a_point():
    from ragsec.experiments import retrieval_experiment

    def recall(tenants, factor):
        return retrieval_experiment(tenants, modes=[("post_overfetch", factor)])[0]["recall"]

    assert recall(3, 4) == 1.0      # enough headroom with three tenants
    assert recall(12, 4) < 0.6      # the same factor is not enough with twelve
    assert recall(12, 16) == 1.0    # and the fix costs fetching 16x the candidates


def test_partition_and_prefilter_agree(w):
    for p in w.principals[:8]:
        a = retrieve(w, p, "quarterly budget forecast", 4, "pre")
        b = retrieve(w, p, "quarterly budget forecast", 4, "partition")
        assert {c.doc_id for c in a.chunks} == {c.doc_id for c in b.chunks}


def test_baked_acl_goes_stale_and_authoritative_does_not():
    w = build_world(3)
    bob = _user(w, "acme", "bob")
    w.revoke("acme-budget-1", {"exec"}, rebake=False)
    stale = retrieve(w, bob, "quarterly budget forecast", 4, "pre", "baked")
    fresh = retrieve(w, bob, "quarterly budget forecast", 4, "pre", "authoritative")
    assert "acme-budget-1" in {c.doc_id for c in stale.chunks}
    assert "acme-budget-1" not in {c.doc_id for c in fresh.chunks}


def test_deleted_document_is_unretrievable_after_tombstone():
    w = build_world(3)
    carol = _user(w, "acme", "carol")
    w.delete("acme-pricing-1")
    for mode in MODES:
        r = retrieve(w, carol, "enterprise pricing memo", 8, mode, "baked")
        assert "acme-pricing-1" not in {c.doc_id for c in r.chunks}, mode


def test_unknown_mode_and_source_are_errors(w):
    p = w.principals[0]
    with pytest.raises(ValueError):
        retrieve(w, p, "x", 4, "magic")
    with pytest.raises(ValueError):
        retrieve(w, p, "x", 4, "pre", "psychic")


def test_shared_idf_is_a_cross_tenant_side_channel_and_partitions_are_not():
    from ragsec.experiments import idf_side_channel
    s = idf_side_channel()
    assert abs(s["shared"][0] - s["shared"][1]) > 1.0
    assert s["partition"][0] == s["partition"][1]
