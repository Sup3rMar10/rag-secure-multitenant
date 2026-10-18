# rag-secure-multitenant

![ci](https://github.com/Sup3rMar10/rag-secure-multitenant/actions/workflows/ci.yml/badge.svg)

Security and multi-tenancy for RAG: **who may retrieve what**, **what a cache is allowed to remember**, **what a poisoned document can make the model do**, and **what happens the moment access is revoked**. Every insecure shortcut is implemented next to the hardened version and measured, so each claim below is a number you can reproduce with `python -m ragsec demo`.

> **What is verified, and what is not.** 162 tests pass locally on Windows with Python 3.13 (stdlib only). The GitHub Actions workflow (Linux, Python 3.11 and 3.13) is included but had not run when this README was written, so Python 3.11 is untested. I broke the code on purpose 16 times (pre-filter removed, tenant check removed, cache key without fingerprint or epoch, delimiter escaping removed, tombstone statistics, scanner, tool policy, canary normalisation, audit verification, baked ACLs, ...) and the suite failed every time; one mutation first *survived* (canary normalisation) and led to two new tests. **There is no real model, no real vector store and no real tenant data here.** The "LLM" is a deterministic stand-in (`ragsec/agent.py`) that is deliberately gullible, the corpus is ~200 synthetic chunks about fictional companies (no card numbers, no health data, no real people; a test scans for them), and retrieval is BM25 only. The injection detector was written by me, and the "held-out" attacks were written by me too, so held-out is *less* independent than the word suggests.

## The threat model in one paragraph

Several customers (tenants) share one RAG service. Within a tenant, documents have group ACLs (`finance`, `exec`, `eng`, or `all`). A user must never see a chunk they cannot read, not through retrieval, not through a cache, not through a stale index, not through a model that was talked into repeating it. Separately, any document a user can edit (a wiki page) is attacker-controlled text that will be pasted into the model's prompt.

## 1. Where the permission check goes

264 queries (every user asks 22 questions), 3 tenants, k = 4. *Leak* = a chunk the asker cannot read reached the model's context. *Recall* = share of the readable relevant documents that were retrieved.

| Retrieval mode | Leak | Mean chunks returned | Recall@4 | "N results" reveals hidden docs |
|---|---|---|---|---|
| none (shared index, no filter) | **100%** | 4.00 | 0.444 | no |
| tenant filter only (forgets groups) | **40.9%** | 4.00 | 1.000 | no |
| post-filter (top-k first, then drop) | 0% | **0.79** | **0.444** | **yes, 100%** |
| post-filter, fetch 4x candidates | 0% | 2.36 | 1.000 | yes, 100% |
| **pre-filter** (permission check inside the index) | 0% | 2.73 | 1.000 | no |
| per-tenant partition + group pre-filter | 0% | 2.73 | 1.000 | no |

Post-filtering is *correct* but starves: the other tenants' near-identical documents take the top-k slots and are then thrown away, leaving 0.79 of 4 chunks on average. (Its recall equals the unfiltered recall by construction: it returns the unfiltered top-4 minus what you may not see.) Fetching more candidates repairs it only while tenants are few:

| Tenants sharing the index | post | post, 4x | post, 16x | pre-filter |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 0.444 | 1.000 | 1.000 | 1.000 |
| 6 | 0.222 | 0.889 | 1.000 | 1.000 |
| 12 | 0.111 | 0.444 | 1.000 | 1.000 |

Two further reasons not to post-filter. A "showing 2 of 7 results" message (or any total) tells a user that restricted documents match their query. And a **shared index leaks corpus statistics**: when another tenant ingested 30 pages, an unchanged acme query's top BM25 score moved from 10.95 to 6.99 on the shared index, and stayed at 10.75 on a per-tenant partition. Partitions remove that side channel; a pre-filter on a shared index does not.

## 2. The cache key is a security boundary

Same workload, pre-filter retrieval, scanner off so the cache is the only thing under test. Questions are matched by token overlap (Jaccard >= 0.8), so a reworded question hits.

| Cache key | Hit rate | Answers containing another **tenant's** chunks | Answers containing a chunk the asker's **group** cannot read |
|---|---|---|---|
| no cache | 0% | 0% | 0% |
| question only | 95.8% | **48.5%** | 6.8% |
| question + tenant | 87.5% | 0% | **20.5%** |
| question + tenant + ACL fingerprint + epoch | 50.0% | 0% | 0% |

The safe key costs about half the hit rate. That is the honest trade: the privileged users warm the cache, and everyone else must miss. (`epoch` is the tenant's ACL version; see section 4 for why it is needed.)

## 3. Prompt injection from retrieved documents

Detector: a transparent rule list (override phrases, forged role markers, exfiltration verbs, weak signals that must add up), run after NFKC folding, zero-width stripping and a small homoglyph map. Defences compared: **detect + quarantine/neutralise**, **delimit** (wrap each chunk in `<untrusted-doc>`, with `<`/`>` escaped so a chunk cannot close the block), and a **tool policy** (a read-only assistant may not call `send_email`, whatever the prompt says).

| Split | Attacks | Benign (many on purpose instruction-like) | Precision | Recall | False positives |
|---|---|---|---|---|---|
| dev (rules tuned here) | 18 | 12 | 0.95 | 1.00 | `b04` "System: Linux is the supported platform..." |
| **held-out (first run, detector not changed afterwards)** | 18 | 12 | **0.77** | **0.56** | `b15`, `b16`, `b24` |

Held-out **misses** (honest bypasses): paraphrase (H02, H07), another language (H04, French), base64 (H05), letter-spaced text (H06), a framing claim (H11, "BEGIN ADMIN OVERRIDE"), "note to AI assistants" (H12), a priority claim (H18). The detector is a speed bump, not a wall.

Attack success rate (ASR) is measured end to end against the toy agent, which **is more capable than the detector** (it understands paraphrase, French, base64 and spaced letters). Every one of the 36 attacks hijacks the undefended agent; a test enforces that, because a low ASR means nothing if the attack never worked.

| Defence | ASR dev | ASR held-out | Held-out, tool policy on | Legit questions still answered | ... of the instruction-like ones |
|---|---|---|---|---|---|
| **agent honours delimiters** | | | | | |
| none | 100% | 100% | 89% | 100% | 100% |
| delimit only | 0% | 0% | 0% | 100% | 100% |
| delimit, **no escaping** | 6% | 6% | 6% | 100% | 100% |
| detect + quarantine | 0% | 44% | 44% | 83% | 79% |
| detect + neutralise | 6% | 44% | 44% | 83% | 79% |
| neutralise + delimit | 0% | 0% | 0% | 83% | 79% |
| **agent ignores delimiters** | | | | | |
| delimit only | 94% | 100% | 89% | 100% | 100% |
| detect + neutralise | 6% | 44% | 44% | 83% | 79% |
| neutralise + delimit | 6% | 44% | 44% | 83% | 79% |

How to read it: **delimiters carry the result, and they depend on the model honouring them**, which is an assumption (`respects_delimiters=True`) I cannot verify without a real model. Remove it and the detector alone lets 44% of held-out attacks through. The detector costs real users: 17% of legitimate questions go unanswered because their documents sound like instructions ("System: the billing service depends on the ledger service"). Escaping matters: without it, a chunk that contains `</untrusted-doc>` breaks out of its block (D10, H17). The tool policy removes the exfiltration attacks outright (89% held-out ASR is the 16 hijack-style attacks; the 2 `send_email` ones drop to 0).

## 4. Revocation and deletion

A user who may read finance documents asks (answer is cached); their access is then removed; they ask again. *Leak* = the second answer contains a document they can no longer read.

| Change | ACL source at query time | Cache key | Index re-baked / tombstoned | Leak |
|---|---|---|---|---|
| revoke | copy baked into the chunk | none | no | **100%** |
| revoke | copy baked into the chunk | none | yes | 0% |
| revoke | authoritative store | none | no | 0% |
| revoke | authoritative store | question + tenant + fingerprint, **no epoch** | no | **100%** |
| revoke | authoritative store | question + tenant + fingerprint + epoch | no | 0% |
| delete | copy baked into the chunk | none | no | **100%** |
| delete | copy baked into the chunk | none | yes | 0% |
| delete | authoritative store | none | no | 0% |
| delete | authoritative store | no epoch | yes | **100%** |
| delete | authoritative store | with epoch | yes | 0% |

Two independent stale copies exist: the ACL baked into index metadata, and the cache. The fix for both is the same idea: check the authoritative store at query time, and make every permission change bump a version that is part of the cache key. A tombstone also removes the document from the BM25 statistics, so scores equal those of an index that never held it (tested against a freshly built index).

## 5. End to end: seven variants, the same 264 queries

Each tenant also has a public support page with an injected payload (an override, an exfiltration and a forged role marker). *Poison in ctx* = queries where that chunk reached the model; hijack/exfil rates are over those queries only (naive and post-filter never retrieved it, so they show n/a).

| Variant | Context leak | Cross-tenant | Canary visible in answer | Scanner blocked | Recall@4 | Poison in ctx | Hijack | Exfil |
|---|---|---|---|---|---|---|---|---|
| naive (no filter, question-only cache, no defence) | 100% | 100% | 56.1% | 0 | 0.427 | 0 | n/a | n/a |
| tenant filter only | 40.9% | 0% | 40.9% | 0 | 1.000 | 24 | 67% | 33% |
| post-filter | 0% | 0% | 0% | 0 | 0.427 | 0 | n/a | n/a |
| pre-filter + baked ACL + epoch-less cache | 0% | 0% | 0% | 0 | 1.000 | 24 | 67% | 33% |
| **hardened** (pre-filter, authoritative ACL, scoped cache, detect + neutralise + delimit, scanner, tool policy) | 0% | 0% | 0% | 0 | 1.000 | 24 | **0%** | **0%** |
| hardened + per-tenant partitions | 0% | 0% | 0% | 0 | 1.000 | 24 | 0% | 0% |
| hardened, agent ignores delimiters | 0% | 0% | 0% | 0 | 1.000 | 24 | 0% | 0% |

The last row is 0% only because these three payloads are ones the detector catches. With **evasive** payloads (held-out H02, H06, H07, which the detector misses) the same matrix gives: hardened (agent honours delimiters) **0%** hijack, hardened (agent ignores delimiters) **100%** hijack. That is the residual risk, stated as a number.

### Canaries and the audit log

- **Canary scanner.** A unique token planted in a restricted document is searched for (case, spacing, zero-width, full-width and homoglyph tricks tolerated) in every outgoing answer. It only knows documents that carry a canary: on the leaky tenant-filter variant it caught 0% / 15% / 52% / 100% of leaking queries at 0% / 25% / 50% / 100% canary coverage. It is a tripwire and an incident detector, not an access control. A base64-encoded canary is missed (a test pins that). Canary tokens are excluded from BM25 scoring: otherwise canary documents are longer, rank lower, and leaks look rarer than they are (I found that the hard way, see below).
- **Audit log.** Hash-chained; stores chunk and document ids, never text. After the naive variant "leaked" `acme-pricing-2`, `who_saw()` named 12 users, 10 of whom were not entitled to it; editing, deleting or reordering an entry breaks `verify()`. The chain is tamper-*evident*, not tamper-*proof*: someone who can rewrite the whole chain defeats it unless the head hash is anchored somewhere else.

## What this does not show

- **No real model.** The agent obeys injected instructions by rule. Whether a real model honours `<untrusted-doc>` blocks, and how often, is the biggest open question; the table above gives both answers. The LLM adapters one would add are untested live (none are included).
- **BM25 only.** With dense vectors the same issues exist (an ANN index with a filter can also starve or leak through shared graph structure), but I did not measure them.
- **Small and synthetic.** 3 to 12 tenants, ~200 to 800 chunks, one author. Results are directional; no latency or throughput was measured.
- **Simple ACL model.** Flat groups per document. No inheritance, no per-chunk ACLs, no attribute rules, no time-boxed grants.
- **Detector scores come from my own corpora.** Dev and held-out were written by the same person in the same sitting. I saw the held-out detector results during development and did not change the detector afterwards; I did change the *agent* once so that attack H18 actually worked against it.
- **Cache side channel.** Users with the same ACL fingerprint share entries, so one can learn that another asked a similar question (hit versus miss, or latency). Not mitigated.
- **Existence oracles beyond the ones tested.** Timing, error messages and "did you mean" are not modelled.

## Run it

```
python -m ragsec demo         # everything above, ~6 seconds
python -m ragsec retrieval    # section 1
python -m ragsec cache        # section 2
python -m ragsec injection    # section 3
python -m ragsec lifecycle    # section 4
python -m ragsec matrix       # section 5, canary coverage, audit
python -m pytest              # 162 tests, ~40 seconds
```

## Layout

| File | Role |
|---|---|
| `ragsec/acl.py` | principals, document ACLs, the one `allows` rule, per-tenant epoch |
| `ragsec/index.py` | BM25 with in-index pre-filter, exact tombstones, existence counter |
| `ragsec/retrieval.py` | the six permission modes, baked versus authoritative ACL |
| `ragsec/cache.py` | semantic cache with four key scopes |
| `ragsec/canary.py`, `ragsec/normalise.py` | canary registry and scanner; Unicode / obfuscation folding |
| `ragsec/injection.py` | detector, neutraliser, delimiting with escaping |
| `ragsec/agent.py` | the gullible stand-in agent and the tool policy |
| `ragsec/audit.py` | hash-chained audit log |
| `ragsec/pipeline.py` | `SecureRAG`: cache, retrieval, defence, agent, tool policy, scanner, audit |
| `ragsec/corpus.py`, `ragsec/attacks.py` | synthetic tenants and the labelled attack/benign corpus |
| `ragsec/experiments.py` | every table above |

## Things that went wrong while building it (and what they taught)

- The first attack-success check counted a marker that was merely *quoted* in the answer as a hijack: "delimit only" looked 67% vulnerable against an agent that honoured delimiters. ASR now requires that the agent actually obeyed (and a mutation test guards it).
- Canaries made documents longer, so they ranked lower and the naive variant showed 0 canary leaks while leaking on every query. Scoring now ignores canary tokens.
- The answer text quoted a filler paragraph without the canary, hiding leaks. Chunks now carry the document header, as contextual-header chunkers do.
- Attack H18 did not work against the undefended agent (its cue came after the marker), so its "blocked" result would have been meaningless. A strawman test now requires every attack to succeed undefended.
- The agent's block parser was case-sensitive, which made the case-variant breakout H17 impossible; a real model would read `</UNTRUSTED-DOC >` as a closing tag, so the parser was made tolerant.

MIT licensed.
