"""Tokenising helpers shared by the index, the cache and the leak scanner."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
# Canary tokens are excluded from scoring: planting one must not change a document's rank,
# otherwise canary documents rank lower and leaks look rarer than they are.
_CANARY = re.compile(r"CANARY-[0-9a-f]{10}")
STOP = frozenset(
    "a an the of to in on for and or is are was were be by with at as it this that "
    "what how do does we our us from".split()
)


def stem(token: str) -> str:
    """Crude plural folding: enough to make 'prices' and 'price' collide."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = _CANARY.sub(" ", text)
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP]


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
