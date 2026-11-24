"""Deterministic, regex-based structured-math problem router (Phase 3).

This is NOT an LLM classifier and not a natural-language understanding
system — it recognizes a small, explicit, auditable set of imperative
computation commands ("differentiate 2*x^2", "integrate cos(x)",
"5 choose 2") and routes only those to the deterministic SymPy solvers
in `axiom/verification/engine.py`, bypassing the evidence-grounded
Meta/RAG path entirely for them.

Deliberately prefix-anchored and narrow: a natural interrogative
question like "What is the derivative of sin(x)?" does **not** match.
That phrasing could just as easily be a conceptual question ("explain
what a derivative means") as a request to compute one specific value —
distinguishing those from free text is a real natural-language
understanding problem this router does not attempt to solve. Recall is
deliberately sacrificed for precision: a missed computation request
still gets a reasonable (if unverified) answer from the existing
evidence-grounded path; a wrongly-intercepted conceptual question would
instead get a terse computed number with no explanation, which is a
worse failure mode. See `docs/jee-solving.md` for the exact grammar and
the reasoning behind this scope boundary.
"""

import re
from dataclasses import dataclass
from typing import Literal

ProblemKind = Literal["derivative", "integral", "ncr", "npr"]


@dataclass
class StructuredMathProblem:
    kind: ProblemKind
    expression: str = ""
    variable: str = "x"
    n: int | None = None
    r: int | None = None


_DIFFERENTIATE_RE = re.compile(r"^\s*differentiate\s+(.+?)\s*[.?!]*\s*$", re.IGNORECASE)
_DDX_RE = re.compile(r"^\s*d\s*/\s*d([a-zA-Z])\s*\(?\s*(.+?)\s*\)?\s*[.?!]*\s*$", re.IGNORECASE)
_INTEGRATE_RE = re.compile(r"^\s*integrate\s+(.+?)\s*[.?!]*\s*$", re.IGNORECASE)
_NCR_WORD_RE = re.compile(r"^\s*(\d{1,3})\s*choose\s*(\d{1,3})\s*[.?!]*\s*$", re.IGNORECASE)
_NCR_FUNC_RE = re.compile(r"^\s*C\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)\s*[.?!]*\s*$")
_NPR_FUNC_RE = re.compile(r"^\s*P\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)\s*[.?!]*\s*$")


def match_structured_math_problem(question: str) -> StructuredMathProblem | None:
    """Returns `None` for anything that doesn't match one of the
    explicit command patterns above — the caller's existing behavior
    (the evidence-grounded path) is then completely unchanged."""
    text = question.strip()

    match = _DDX_RE.match(text)
    if match:
        return StructuredMathProblem(
            kind="derivative", expression=match.group(2), variable=match.group(1)
        )

    match = _DIFFERENTIATE_RE.match(text)
    if match:
        return StructuredMathProblem(kind="derivative", expression=match.group(1))

    match = _INTEGRATE_RE.match(text)
    if match:
        return StructuredMathProblem(kind="integral", expression=match.group(1))

    match = _NCR_WORD_RE.match(text) or _NCR_FUNC_RE.match(text)
    if match:
        return StructuredMathProblem(kind="ncr", n=int(match.group(1)), r=int(match.group(2)))

    match = _NPR_FUNC_RE.match(text)
    if match:
        return StructuredMathProblem(kind="npr", n=int(match.group(1)), r=int(match.group(2)))

    return None
