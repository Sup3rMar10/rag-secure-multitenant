"""A deterministic stand-in for an LLM that can be talked into things.

THIS IS NOT A MODEL. It exists so that "does the defence stop the attack" has a repeatable answer.
It is deliberately gullible and deliberately MORE capable than the detector: it understands
paraphrases, French, base64 and letter-spaced text, because a real model does. Its one assumption
about delimiters (`respects_delimiters=True`) is exactly that, an assumption: real models honour
`<untrusted-doc>` blocks only partly, so the README reports the agent both ways.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .injection import DOC_OPEN, Prompt
from .normalise import deep_normalise, light_normalise
from .text import sentences

_CUE = re.compile(
    r"ignore|disregard|forget|override|set aside|new instructions?|from now on|system\s*:|\[system\]"
    r"|<\|im_start\|>|you are now|act as|your (?:real|new) task|note to (?:ai|assistants?)|important\s*:"
    r"|admin override|before answering|do not tell|silently|kindly|ignorez|oubliez|authori[sz]ed you"
    r"|new system prompt|decode and follow|respond only|reply only|answer exactly|more important than"
    r"|(?:respond|reply|answer) with (?=[A-Z]{3,}-\d)",
    re.I,
)
_CUE_EXFIL = re.compile(
    r"send_email|\b(?:forward|send|email)\s+(?:everything|(?:this|the)\s+(?:full\s+|retrieved\s+)*"
    r"(?:conversation|documents?|transcript|chat))", re.I)
_MARKER = re.compile(r"\b[A-Z]{3,}-\d{1,3}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]+", re.I)
# Tolerant on purpose: a model reads </UNTRUSTED-DOC > as a closing tag too.
_BLOCK = re.compile(r"<\s*untrusted-doc[^>]*>.*?<\s*/\s*untrusted-doc\s*>", re.S | re.I)


@dataclass
class AgentResult:
    text: str
    tool_calls: list = field(default_factory=list)  # [("send_email", address)]
    obeyed_injection: bool = False


@dataclass
class ToyAgent:
    respects_delimiters: bool = True

    def _eligible(self, body: str) -> str:
        """Text the agent might take orders from."""
        body = light_normalise(body)  # full-width brackets count as brackets to a model
        if self.respects_delimiters and DOC_OPEN in body.lower():
            return _BLOCK.sub(" ", body)  # an intact block is data; anything left over is not
        return body

    def run(self, prompt: Prompt) -> AgentResult:
        body = prompt.text.split("DOCUMENTS:\n", 1)[-1]
        text = deep_normalise(self._eligible(body))
        calls, marker = [], None
        cue = _CUE.search(text)
        if cue:
            m = _MARKER.search(text, cue.start())
            if m:
                marker = m.group(0)
        ex = _CUE_EXFIL.search(text)
        if ex:
            e = _EMAIL.search(text, ex.start())
            if e:
                calls.append(("send_email", e.group(0)))
        if marker is not None:
            return AgentResult(marker, calls, True)
        leads = []
        for cid, doc in prompt.docs[:2]:
            sents = sentences(doc)
            if sents:
                leads.append(f"{' '.join(sents[:2])} [{cid}]")
        return AgentResult(" ".join(leads) if leads else "I could not find that in the documents.", calls, bool(calls))


@dataclass(frozen=True)
class ToolPolicy:
    """Defence in depth: a read-only assistant has no business sending email, whatever the prompt says."""
    allowed: frozenset = frozenset()

    def apply(self, calls):
        executed = [c for c in calls if c[0] in self.allowed]
        blocked = [c for c in calls if c[0] not in self.allowed]
        return executed, blocked
