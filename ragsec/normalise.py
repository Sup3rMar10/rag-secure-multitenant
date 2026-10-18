"""Unicode and obfuscation folding. Case is preserved; callers use re.I when they need it."""
from __future__ import annotations

import base64
import re
import unicodedata

# Zero-width and soft-hyphen characters attackers insert inside words.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# A handful of Cyrillic/Greek letters that render like Latin ones. NOT a full confusables table.
_HOMOGLYPH = {
    ord(c): lat
    for c, lat in zip("аеорсхіуѕАЕОРСХІ", "aeopcxiysAEOPCXI")
}

_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def light_normalise(text: str) -> str:
    """What the DETECTOR sees: NFKC (folds full-width letters), invisible characters removed, homoglyphs folded."""
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_HOMOGLYPH)


_SPACED = re.compile(r"(?:\S ){3,}\S(?:\s{2,}(?:\S ){3,}\S)*")


def _join_spaced(m: re.Match) -> str:
    words = re.split(r"\s{2,}", m.group(0))
    return " ".join(w.replace(" ", "") for w in words)


def deep_normalise(text: str) -> str:
    """What the toy AGENT understands: everything light_normalise does, plus letter-spaced text and base64.

    It is deliberately more capable than the detector: real models decode and un-obfuscate more than
    a regex pipeline does, and a defence that is tested only against an agent as dumb as itself proves nothing.
    """
    text = light_normalise(text)
    text = _SPACED.sub(_join_spaced, text)
    extras = []
    for blob in _B64.findall(text):
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            decoded = raw.decode("ascii")
        except Exception:
            continue
        if decoded.isprintable():
            extras.append(decoded)
    if extras:
        text = text + "\n" + "\n".join(extras)
    return text


def flat(text: str) -> str:
    """Letters and digits only, lower-case. Used to find a canary however it was spaced or punctuated."""
    text = light_normalise(text).lower()
    return re.sub(r"[^a-z0-9]", "", text)
