from typing import Literal

from pydantic import BaseModel, Field

VerificationStatus = Literal[
    "verified_correct", "verified_incorrect", "incomplete", "unsupported", "indeterminate"
]
Category = Literal["linear", "quadratic", "unsupported"]


class VerifyRequest(BaseModel):
    equation: str = Field(min_length=1, max_length=200)
    variable: str = Field(default="x", min_length=1, max_length=1, pattern=r"^[A-Za-z]$")
    solutions: list[str] = Field(default_factory=list, max_length=20)


class CandidateResponse(BaseModel):
    value: str
    valid: bool | None


class VerifyResponse(BaseModel):
    """`status` discriminates the verdict:

    - `"verified_correct"` — every proposed value satisfies the original
      equation, and the set is exactly complete (nothing real is missing).
    - `"verified_incorrect"` — at least one proposed value does not
      satisfy the equation, or a false "no real solutions" claim was
      made (an empty `solutions` list against a nonempty real solution
      set).
    - `"incomplete"` — every proposed value is individually valid, but
      the set is missing at least one real solution.
    - `"unsupported"` — the equation (or a proposed value) falls outside
      this engine's scope: not single-variable, not linear/quadratic,
      not polynomial, or otherwise unparseable under the restricted
      grammar. See `reason`.
    - `"indeterminate"` — the problem is in scope, but SymPy could not
      exactly decide the real solution set or a candidate's validity.
      Never treated as correct or incorrect.

    `complete` is only meaningful once candidate validity has already
    been established; `None` when it doesn't apply (e.g. `"unsupported"`).
    `deterministic` is `False` only for `"indeterminate"` results —
    every other status was reached by exact SymPy computation, never a
    floating-point approximation or an LLM judgment."""

    status: VerificationStatus
    category: Category
    domain: Literal["real"]
    deterministic: bool
    reason: str | None = None
    detail: str
    complete: bool | None = None
    candidates: list[CandidateResponse] = []
