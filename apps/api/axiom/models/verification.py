from typing import Annotated, Literal

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


TransitionKind = Literal[
    "equivalent",
    "narrowing",
    "broadening",
    "partial_overlap",
    "incompatible",
    "unsupported",
    "indeterminate",
]
StepCheckStatus = Literal[
    "all_steps_equivalent",
    "valid_but_incomplete",
    "invalid_step_found",
    "unsupported",
    "verification_incomplete",
]

_BoundedEquation = Annotated[str, Field(min_length=1, max_length=200)]


class CheckWorkRequest(BaseModel):
    """`solutions` distinguishes "no final answer submitted" (omitted,
    the default `None` — `final_answer` in the response is then `None`
    too) from "the student is claiming the equation has no real
    solutions" (an explicit empty list `[]`), the same convention
    `VerifyRequest` already uses."""

    equation: str = Field(min_length=1, max_length=200)
    variable: str = Field(default="x", min_length=1, max_length=1, pattern=r"^[A-Za-z]$")
    steps: list[_BoundedEquation] = Field(default_factory=list, max_length=15)
    solutions: list[str] | None = Field(default=None, max_length=20)


class StepResult(BaseModel):
    index: int
    equation: str
    category: Category
    transition: TransitionKind
    detail: str


class CheckWorkResponse(BaseModel):
    """`status` summarizes the whole chain:

    - `"all_steps_equivalent"` — every step preserved the exact real
      solution set of the original equation.
    - `"valid_but_incomplete"` — no provably incorrect step was found,
      but at least one step narrowed or broadened the previous solution
      set instead of preserving it exactly.
    - `"invalid_step_found"` — at least one step was provably
      incompatible with the previous one, or dropped a genuine solution
      while introducing an unverified one (`"partial_overlap"`, which is
      treated with the same severity as `"incompatible"` — see
      `first_invalid_step_index`).
    - `"unsupported"` — the *original* equation itself is outside the
      supported grammar; no steps could be checked at all (`steps` is
      empty).
    - `"verification_incomplete"` — at least one individual step
      couldn't be parsed, classified, or exactly solved, breaking
      verification continuity at that point; later steps may still have
      been checked against each other once a decidable step reappeared.

    `chain_preserves_solution_set` is `True` only if every step was
    exactly equivalent, `False` as soon as any step is confirmed
    non-equivalent (regardless of unrelated gaps elsewhere), and `None`
    when the only problems encountered were continuity gaps with no
    confirmed non-equivalence either way. `final_answer`, when present,
    is always computed against the *original* equation (step 0), never
    against the student's last written step — it is a separate
    dimension from step-reasoning correctness; see
    `docs/verification.md`."""

    status: StepCheckStatus
    steps: list[StepResult]
    first_invalid_step_index: int | None
    first_non_equivalent_step_index: int | None
    chain_preserves_solution_set: bool | None
    deterministic: bool
    detail: str
    final_answer: VerifyResponse | None = None
