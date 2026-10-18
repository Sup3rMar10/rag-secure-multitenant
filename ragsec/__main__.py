"""python -m ragsec demo | retrieval | cache | injection | lifecycle | matrix"""
from __future__ import annotations

import sys

from . import experiments as ex
from .attacks import HELDOUT_ATTACKS


def _table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def _rates(r):
    if not r["exposed_queries"]:
        return ["n/a", "n/a"]   # the poisoned page never reached the model in this variant
    return [f"{r['hijack_pct']:.0f}", f"{r['exfil_pct']:.0f}"]


def retrieval() -> None:
    print("== 1. Applying permissions to retrieval (3 tenants, k=4, 264 queries) ==")
    rows = ex.retrieval_experiment()
    print(_table(["mode", "leak%", "mean results", "recall@4", "existence oracle%"],
                 [[r["mode"], f"{r['leak_pct']:.1f}", f"{r['mean_results']:.2f}", f"{r['recall']:.3f}",
                   f"{r['oracle_pct']:.1f}"] for r in rows]))
    print("\nrecall@4 of post-filtering as tenants share the index:")
    sv = ex.starvation_vs_tenants()
    print(_table(["tenants", "post", "post x4", "post x16", "pre"],
                 [[r["tenants"], f"{r['post']:.3f}", f"{r['post_overfetch(x4)']:.3f}",
                   f"{r['post_overfetch(x16)']:.3f}", f"{r['pre']:.3f}"] for r in sv]))
    s = ex.idf_side_channel()
    print("\ntop score of an acme query before/after ANOTHER tenant ingests 30 pages:")
    for k, (b, a) in s.items():
        print(f"  {k:9s} {b:.4f} -> {a:.4f}  ({'CHANGED' if abs(a - b) > 1e-9 else 'unchanged'})")


def cache() -> None:
    print("\n== 2. Semantic cache key (pre-filter retrieval, scanner off) ==")
    rows = ex.cache_experiment()
    print(_table(["cache key", "hit%", "cross-tenant leak%", "cross-group leak%"],
                 [[r["cache"], f"{r['hit_pct']:.1f}", f"{r['cross_tenant_pct']:.1f}",
                   f"{r['cross_group_pct']:.1f}"] for r in rows]))


def injection() -> None:
    print("\n== 3. Prompt-injection defence ==")
    d = ex.detection_experiment()
    print(_table(["split", "attacks", "benign", "precision", "recall", "F1", "false positives"],
                 [[s, v["attacks"], v["benign"], f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}",
                   ",".join(v["false_positives"]) or "-"] for s, v in d.items()]))
    print("held-out misses:", ", ".join(f"{i}({k})" for i, k in d["heldout"]["missed"]))
    print()
    rows = ex.asr_experiment()
    for respects in (True, False):
        print(f"\nagent honours delimiters = {respects}")
        print(_table(["defence", "ASR dev%", "ASR held-out%", "held-out + tool policy%", "utility%", "utility (hard)%"],
                     [[r["defence"], f"{r['asr_dev']:.0f}", f"{r['asr_heldout']:.0f}",
                       f"{r['asr_heldout_with_tool_policy']:.0f}", f"{r['utility']:.0f}", f"{r['utility_hard']:.0f}"]
                      for r in rows if r["agent_respects_delimiters"] == respects]))


def lifecycle() -> None:
    print("\n== 4. Revocation and deletion (leak% when access was removed AFTER the answer was cached) ==")
    rows = ex.lifecycle_experiment()
    print(_table(["change", "ACL source", "cache key", "re-baked/tombstoned", "leak%"],
                 [[r["kind"], r["acl_source"], r["cache"],
                   r["rebake"] if r["kind"] == "revoke" else r["tombstone"], f"{r['leak_pct']:.0f}"] for r in rows]))


def matrix() -> None:
    print("\n== 5. End to end, seven variants, same 264 queries, poisoned support page in every tenant ==")
    rows = ex.matrix_experiment()
    print(_table(["variant", "ctx leak%", "x-tenant%", "answer leak%", "scanner blocked", "recall@4",
                  "poison in ctx", "hijack%", "exfil%"],
                 [[r["variant"], f"{r['ctx_leak_pct']:.1f}", f"{r['cross_tenant_pct']:.1f}",
                   f"{r['answer_leak_pct']:.1f}", r["scanner_blocked"], f"{r['recall']:.3f}",
                   r["exposed_queries"], *_rates(r)] for r in rows]))
    evasive = [a.text for a in HELDOUT_ATTACKS if a.id in ("H02", "H06", "H07")]
    print("\nsame matrix, poisoned page carries EVASIVE payloads (held-out H02, H06, H07 that the detector misses):")
    rows = ex.matrix_experiment(evasive, [ex.NAIVE, ex.HARDENED, ex.HARDENED_UNTRUSTED_DELIMS])
    print(_table(["variant", "poison in ctx", "hijack%", "exfil%"],
                 [[r["variant"], r["exposed_queries"], *_rates(r)] for r in rows]))
    print("\ncanary scanner coverage (leaky tenant-filter variant, scanner on):")
    print(_table(["canary rate", "leaky queries", "caught", "caught%"],
                 [[r["canary_rate"], r["leaky_queries"], r["caught"], f"{r['caught_pct']:.0f}"]
                  for r in ex.scanner_coverage()]))
    b = ex.audit_blast_radius()
    print(f"\naudit: after a leak of {b['doc']} in the naive variant, who_saw() names {b['viewers']} users, "
          f"{b['not_entitled']} of them not entitled; chain verifies = {b['chain_ok']} ({b['entries']} entries)")


COMMANDS = {"retrieval": retrieval, "cache": cache, "injection": injection, "lifecycle": lifecycle, "matrix": matrix}


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "demo"
    if cmd == "demo":
        for fn in COMMANDS.values():
            fn()
        return 0
    if cmd in COMMANDS:
        COMMANDS[cmd]()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
