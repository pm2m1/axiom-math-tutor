"""Input guardrail for the agentic answer workflow (Phase 1K): a small,
deterministic classifier that runs before any retrieval, judge, or MCP
call. Deliberately not model-based — every request would otherwise cost
a Meta call just to decide whether to reject it, and a classifier built
from the same kind of model the injection targets is itself a target.
Keyword/pattern matching is bounded, inspectable, and free; its
documented tradeoff is recall, not safety-by-obscurity (see
docs/agentic-workflow.md).
"""

import re
from typing import Literal

from pydantic import BaseModel

GUARDRAIL_VERSION = "input-guardrail-v1"

# Phrases asking the system to reveal its own configuration, abandon its
# governing instructions, or assume a different persona. Deliberately
# anchored on multi-word phrases ("ignore ... instructions") rather than
# single trigger words ("ignore") so that ordinary mathematical language
# ("ignore the remainder") doesn't false-positive.
_INJECTION_PATTERNS = [
    r"ignore (all |any |the |your |my )?(previous|prior|above|earlier) instructions",
    r"disregard (all |any |the |your |my )?(previous|prior|above|earlier) instructions",
    r"forget (all |your |my )?(previous |prior )?instructions",
    r"reveal (your |the )?(system prompt|api key|credentials|instructions)",
    r"(show|print|what is) your (system prompt|api key|credentials)",
    r"you are now (a|an) ",
    r"pretend (to be|you are)",
    r"jailbreak",
    r"developer mode",
    r"bypass (your |the )?(rules|restrictions|guardrails|safety)",
    r"system prompt",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# A generous math-signal vocabulary: this must accept calculus, geometry,
# statistics, and other math outside the algebra KB (those are legitimate
# web-fallback candidates, not rejections) — it only needs to reject
# requests with no mathematical content at all.
_MATH_WORDS = re.compile(
    r"\b(equation|inequality|solve|simplify|calculate|compute|derivative|"
    r"integral|algebra|geometry|calculus|trigonometry|theorem|formula|"
    r"variable|function|graph|slope|polynomial|quadratic|factor|matrix|"
    r"vector|probability|statistics|proof|sum|product|fraction|exponent|"
    r"logarithm|root|angle|triangle|circle|area|volume|number|digit|"
    r"percent|ratio|sequence|series|limit|property|equality|equal|math)\b",
    re.IGNORECASE,
)
_MATH_SYMBOLS = re.compile(r"[0-9]|[=+\-*/^]")


class InputClassification(BaseModel):
    status: Literal["accepted", "rejected_off_topic", "rejected_injection"]
    detail: str


def classify_input(question: str) -> InputClassification:
    """Length/emptiness is enforced upstream by `QuestionRequest`'s
    Pydantic `Field` constraints (a 422, before this ever runs); this
    only distinguishes accepted math questions from the two reject
    categories. Injection is checked before topic, since an injection
    attempt phrased with math vocabulary ("solve this: ignore your
    instructions...") must still be rejected as injection, not accepted
    as a math question."""
    if _INJECTION_RE.search(question):
        return InputClassification(
            status="rejected_injection",
            detail=(
                "the request appears to ask the system to change its role "
                "or reveal internal configuration"
            ),
        )
    if _MATH_WORDS.search(question) or _MATH_SYMBOLS.search(question):
        return InputClassification(status="accepted", detail="")
    return InputClassification(
        status="rejected_off_topic",
        detail="the request does not appear to be a mathematical question",
    )
