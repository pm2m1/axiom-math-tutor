"""Deterministic SymPy verification engine (Phase 2A, extended in Phase 2B).

Scope, deliberately bounded: single-variable linear and quadratic
equations, rational/integer coefficients, exact real roots (including
simple irrationals expressible via `sqrt`). Nothing here asks an LLM
whether an answer is correct — every verdict is produced by exact SymPy
computation (`Poly`, `solveset` over `S.Reals`, exact symbolic equality)
or is explicitly reported as `"unsupported"`/`"indeterminate"` rather
than guessed. See `docs/verification.md` for the full grammar, status
definitions, and worked examples.

This module never reads a raw string itself — `axiom/verification/parser.py`
is the only boundary that turns text into a SymPy expression, and it is
default-deny. Candidate validity is always checked against the original
`lhs`/`rhs` the equation parsed to, not a transformed/moved-to-one-side
form, per the project's instruction not to trust a candidate merely
because it satisfies an algebraically-transformed equation.

Each step below (`classify`, `solve_real`, `check_candidate`,
`compare_to_solution_set`) is a small, independently testable function;
`verify()` is only the orchestration of those pieces for a single
equation. `check_steps()` (Phase 2B) reuses every one of them to verify
an ordered chain of student equation steps — there is deliberately no
second parser or solver; see `docs/verification.md`'s "Check My Work"
section.
"""

from dataclasses import dataclass, field
from typing import Literal

import sympy
from sympy.polys.polyerrors import PolynomialError

from axiom.verification.parser import (
    ParseError,
    parse_calculus_expression,
    parse_candidate,
    parse_equation,
)

Category = Literal["linear", "quadratic", "unsupported"]
VerificationStatus = Literal[
    "verified_correct", "verified_incorrect", "incomplete", "unsupported", "indeterminate"
]


@dataclass
class CandidateResult:
    value: str
    valid: bool | None  # None only when exact validity could not be decided


@dataclass
class VerificationOutcome:
    status: VerificationStatus
    category: Category
    domain: Literal["real"]
    deterministic: bool
    reason: str | None
    detail: str
    complete: bool | None = None
    candidates: list[CandidateResult] = field(default_factory=list)


def classify(lhs: sympy.Expr, rhs: sympy.Expr, variable: sympy.Symbol) -> Category | None:
    """`None` means unsupported — classification looks only at the
    equation's actual polynomial structure (via SymPy `Poly`, after
    expansion), not the superficial way it was typed. A term that
    algebraically cancels (e.g. `x^2 - x^2 + 2*x - 6`) is classified by
    its real degree, 1 here, not by the highest exponent in the source
    text. An equation that never mentions `variable` at all (a pure
    numeric identity like `5 = 3`) is unsupported — there is nothing to
    solve for."""
    if variable not in lhs.free_symbols and variable not in rhs.free_symbols:
        return None

    moved = sympy.expand(lhs - rhs)
    if moved.free_symbols - {variable}:
        return None

    try:
        poly = sympy.Poly(moved, variable)
    except PolynomialError:
        return None

    degree = poly.degree()
    if degree == -sympy.oo or degree in (0, 1):
        # Degree 0 (or the zero polynomial's degree, -oo) only reaches
        # here because `variable` appeared in the original equation and
        # then cancelled algebraically (e.g. `3*x = 3*x + 1`, a
        # contradiction, or `2*x = 2*x`, an identity) — still a
        # linear-shaped problem, not an unrelated numeric statement. A
        # degree-0 polynomial where `variable` never appeared at all was
        # already rejected above.
        return "linear"
    if degree == 2:
        return "quadratic"
    return None


def solve_real(lhs: sympy.Expr, rhs: sympy.Expr, variable: sympy.Symbol) -> sympy.Set:
    """The exact real solution set of `lhs = rhs`. May return a set this
    engine cannot fully decide (a `ConditionSet` or similar) — callers
    must check `is_decidable_set` before trusting the result as final."""
    moved = sympy.expand(lhs - rhs)
    return sympy.solveset(sympy.Eq(moved, 0), variable, domain=sympy.S.Reals)


def is_decidable_set(solution_set: sympy.Set) -> bool:
    """True for the three shapes this engine can report exactly: no real
    solutions, every real number, or a finite set of concrete (symbol-free)
    numbers. Anything else — a `ConditionSet`, a `Union` involving one, an
    `ImageSet` — is a result SymPy could not fully resolve, and must not be
    mistaken for a finite verified answer."""
    if solution_set == sympy.S.EmptySet or solution_set == sympy.S.Reals:
        return True
    if isinstance(solution_set, sympy.FiniteSet):
        return all(not element.free_symbols for element in solution_set.args)
    return False


def check_candidate(
    lhs: sympy.Expr, rhs: sympy.Expr, variable: sympy.Symbol, value: sympy.Expr
) -> bool | None:
    """Substitutes `value` into the *original* `lhs`/`rhs` (never a
    moved-to-one-side or otherwise transformed form) and decides exact
    equality. `None` means SymPy could not decide — report indeterminate,
    never assume correctness or incorrectness."""
    difference = (lhs - rhs).subs(variable, value)
    decision = difference.equals(0)
    return None if decision is None else bool(decision)


@dataclass
class SetComparison:
    complete: bool
    mismatch: bool  # a candidate validated individually but isn't a true_set member
    missing_count: int


def compare_to_solution_set(
    true_set: sympy.FiniteSet, proposed_values: list[sympy.Expr]
) -> SetComparison:
    """Compares a *validated* list of proposed values (every one of which
    has already passed `check_candidate`) against the complete real
    solution set. `mismatch=True` is a defensive signal, not an expected
    path: every validated candidate should already be a `true_set`
    member, so a mismatch means a representation inconsistency, not a
    normal incomplete/extraneous submission."""
    proposed_set = sympy.FiniteSet(*proposed_values) if proposed_values else sympy.S.EmptySet
    extra = proposed_set - true_set
    missing = true_set - proposed_set
    return SetComparison(
        complete=(missing == sympy.S.EmptySet),
        mismatch=(extra != sympy.S.EmptySet),
        missing_count=len(missing) if isinstance(missing, sympy.FiniteSet) else 0,
    )


def _unsupported(
    detail: str, reason: str, category: Category = "unsupported"
) -> VerificationOutcome:
    return VerificationOutcome(
        status="unsupported",
        category=category,
        domain="real",
        deterministic=True,
        reason=reason,
        detail=detail,
    )


def _indeterminate(
    category: Category, detail: str, reason: str, candidates: list[CandidateResult] | None = None
) -> VerificationOutcome:
    return VerificationOutcome(
        status="indeterminate",
        category=category,
        domain="real",
        deterministic=False,
        reason=reason,
        detail=detail,
        candidates=candidates or [],
    )


def _classification_failure(
    lhs: sympy.Expr, rhs: sympy.Expr, variable: sympy.Symbol
) -> tuple[str, str]:
    """Only called once `classify()` has already returned `None` — re-walks
    the same checks purely to name which one failed, for an honest `reason`/
    `detail` rather than a generic "unsupported"."""
    if variable not in lhs.free_symbols and variable not in rhs.free_symbols:
        return "missing_variable", f"the equation does not reference the variable '{variable}'"

    moved = sympy.expand(lhs - rhs)
    other_symbols = moved.free_symbols - {variable}
    if other_symbols:
        names = ", ".join(sorted(str(s) for s in other_symbols))
        return "multiple_variables", f"the equation contains other symbols: {names}"

    try:
        degree = sympy.Poly(moved, variable).degree()
    except PolynomialError:
        return "not_polynomial", "the equation is not a polynomial in the variable"

    return (
        "unsupported_degree",
        f"degree-{degree} equations are not supported (only linear and quadratic are)",
    )


def verify(equation_text: str, variable: str, solution_texts: list[str]) -> VerificationOutcome:
    try:
        lhs, rhs = parse_equation(equation_text, variable)
    except ParseError as exc:
        return _unsupported(f"could not parse the equation: {exc}", "unparseable_equation")

    var = sympy.Symbol(variable)

    category = classify(lhs, rhs, var)
    if category is None:
        reason, detail = _classification_failure(lhs, rhs, var)
        return _unsupported(detail, reason)

    true_set = solve_real(lhs, rhs, var)
    if not is_decidable_set(true_set):
        return _indeterminate(
            category,
            "the exact real solution set could not be established",
            "unresolved_solution_set",
        )

    parsed_values: list[sympy.Expr] = []
    for text in solution_texts:
        try:
            parsed_values.append(parse_candidate(text))
        except ParseError as exc:
            return _unsupported(
                f"could not parse proposed solution '{text}': {exc}",
                "unparseable_candidate",
                category=category,
            )

    candidates: list[CandidateResult] = []
    undecidable = False
    for text, value in zip(solution_texts, parsed_values, strict=True):
        decision = check_candidate(lhs, rhs, var, value)
        if decision is None:
            undecidable = True
            candidates.append(CandidateResult(value=text, valid=None))
        else:
            candidates.append(CandidateResult(value=text, valid=decision))

    if undecidable:
        return _indeterminate(
            category,
            "could not exactly decide whether a proposed value satisfies the equation",
            "candidate_equality_undecidable",
            candidates,
        )

    any_invalid = any(c.valid is False for c in candidates)

    if true_set == sympy.S.EmptySet:
        if not parsed_values:
            return VerificationOutcome(
                status="verified_correct",
                category=category,
                domain="real",
                deterministic=True,
                reason=None,
                detail="the equation has no real solutions; the empty set is correct",
                complete=True,
            )
        return VerificationOutcome(
            status="verified_incorrect",
            category=category,
            domain="real",
            deterministic=True,
            reason="no_real_solutions",
            detail="the equation has no real solutions, but a value was proposed",
            complete=False,
            candidates=candidates,
        )

    if true_set == sympy.S.Reals:
        if not parsed_values:
            return VerificationOutcome(
                status="verified_incorrect",
                category=category,
                domain="real",
                deterministic=True,
                reason="infinite_solutions",
                detail="the equation holds for every real number; an empty set is incorrect",
                complete=False,
            )
        if any_invalid:
            return VerificationOutcome(
                status="verified_incorrect",
                category=category,
                domain="real",
                deterministic=True,
                reason="invalid_candidate",
                detail="at least one proposed value does not satisfy the equation",
                complete=False,
                candidates=candidates,
            )
        return VerificationOutcome(
            status="incomplete",
            category=category,
            domain="real",
            deterministic=True,
            reason="infinite_solutions",
            detail="the equation holds for every real number; no finite list can be complete",
            complete=False,
            candidates=candidates,
        )

    # `true_set` is a nonempty, finite, fully-decided set of real roots.
    if not parsed_values:
        return VerificationOutcome(
            status="verified_incorrect",
            category=category,
            domain="real",
            deterministic=True,
            reason="missing_solutions",
            detail="real solutions exist, but none were proposed",
            complete=False,
        )

    if any_invalid:
        return VerificationOutcome(
            status="verified_incorrect",
            category=category,
            domain="real",
            deterministic=True,
            reason="invalid_candidate",
            detail="at least one proposed value does not satisfy the equation",
            complete=False,
            candidates=candidates,
        )

    assert isinstance(true_set, sympy.FiniteSet)
    comparison = compare_to_solution_set(true_set, parsed_values)
    if comparison.mismatch:
        # Every candidate passed the direct equation check above, so this
        # would mean a valid root isn't recognized as a `true_set` member —
        # a representation mismatch, not a case to report confidently
        # either way (see module docstring on not trusting transformed
        # equation checks over the original one).
        return _indeterminate(
            category,
            "a validated candidate could not be reconciled with the exact solution set",
            "set_membership_mismatch",
            candidates,
        )

    if comparison.complete:
        return VerificationOutcome(
            status="verified_correct",
            category=category,
            domain="real",
            deterministic=True,
            reason=None,
            detail="the proposed solution set matches the exact real solution set",
            complete=True,
            candidates=candidates,
        )

    return VerificationOutcome(
        status="incomplete",
        category=category,
        domain="real",
        deterministic=True,
        reason="missing_solutions",
        detail=f"{comparison.missing_count} real solution(s) were not proposed",
        complete=False,
        candidates=candidates,
    )


# --- Phase 2B: ordered step-by-step verification ("Check My Work") --------

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

_TRANSITION_DETAIL: dict[TransitionKind, str] = {
    "equivalent": "this step preserves the real solution set",
    "narrowing": (
        "this step removes one or more possible solutions; check whether "
        "you are considering only one branch"
    ),
    "broadening": (
        "this transformation may introduce extraneous solutions; the final "
        "answer must still be checked against the original equation"
    ),
    "partial_overlap": (
        "this step drops at least one previously valid solution while introducing "
        "a different, unverified one — not a legitimate branch or extension, so "
        "this is reported as a provably incorrect transition"
    ),
    "incompatible": "this step changes the solution set incompatibly with the previous equation",
}

_STATUS_DETAIL: dict[StepCheckStatus, str] = {
    "all_steps_equivalent": "every step preserves the original real solution set exactly",
    "valid_but_incomplete": (
        "no provably incorrect step was found, but at least one step narrowed or "
        "broadened the solution set instead of preserving it exactly"
    ),
    "invalid_step_found": "a provably incorrect step was found",
    "unsupported": "the original equation is outside the supported verification grammar",
    "verification_incomplete": (
        "at least one step could not be verified, so the full chain's "
        "correctness could not be established"
    ),
}


MisconceptionLabel = Literal[
    "lost_solution_branch", "introduced_extraneous_root", "sign_error", "general_error"
]


@dataclass
class StepOutcome:
    index: int  # 1-based; the original equation is step 0 and is not in this list
    equation: str
    category: Category
    transition: TransitionKind
    detail: str
    misconception: MisconceptionLabel | None = None


@dataclass
class StepCheckOutcome:
    status: StepCheckStatus
    steps: list[StepOutcome]
    first_invalid_step_index: int | None
    first_non_equivalent_step_index: int | None
    chain_preserves_solution_set: bool | None
    deterministic: bool
    detail: str
    final_answer: VerificationOutcome | None


def classify_transition(old_set: sympy.Set, new_set: sympy.Set) -> TransitionKind:
    """Both sets must already be decidable (see `is_decidable_set`) —
    callers handle the undecidable case separately as `"indeterminate"`,
    this function only compares two known sets exactly:

    - equal → `"equivalent"`.
    - `new_set` a nonempty proper subset of `old_set` → `"narrowing"`
      (some solutions kept, some dropped — a legitimate single-branch
      selection, not an error on its own).
    - `old_set` a proper subset of `new_set` → `"broadening"` (every old
      solution survives, but new ones appeared that still need
      independent checking against the original equation).
    - disjoint (including `new_set` empty while `old_set` isn't) →
      `"incompatible"` — nothing of the previous solution set survived.
    - otherwise, a genuine partial overlap (shares some elements, proper
      subset of neither) → `"partial_overlap"`: this drops at least one
      previously valid solution *and* introduces a different, unverified
      one. Losing a genuine solution is never something a legitimate
      algebraic operation does without a clean subset relationship (pure
      narrowing never adds anything; pure broadening never drops
      anything) — so this is, like `"incompatible"`, reported by
      `check_steps()` as a provably wrong transition, not merely an
      incomplete-but-possibly-legitimate one. The distinct label is kept
      only because the diagnostic (both lost *and* gained) is more
      specific than plain disjointness.
    """
    if old_set == new_set:
        return "equivalent"
    if new_set != sympy.S.EmptySet and (new_set - old_set) == sympy.S.EmptySet:
        return "narrowing"
    if (old_set - new_set) == sympy.S.EmptySet:
        return "broadening"
    if old_set.intersect(new_set) == sympy.S.EmptySet:
        return "incompatible"
    return "partial_overlap"


def classify_misconception(
    old_set: sympy.Set, new_set: sympy.Set, transition: TransitionKind
) -> MisconceptionLabel | None:
    """A deterministic label for a non-equivalent transition, derived
    *only* from facts already established by exact set comparison —
    never a guess at the student's intent or the specific operation
    they performed:

    - `"narrowing"` → `"lost_solution_branch"`: true by the definition
      of narrowing (some solutions were dropped), not an inference.
    - `"broadening"` → `"introduced_extraneous_root"`: true by
      definition (new, unverified solutions appeared).
    - `"incompatible"`/`"partial_overlap"` where every element of
      `old_set` negated equals `new_set` exactly → `"sign_error"`: this
      is **checked**, not assumed — a real, common algebra mistake
      (e.g. forgetting to flip a sign) produces exactly this pattern.
    - Anything else non-equivalent → `"general_error"`: the mathematics
      is confirmed wrong, but nothing here licenses a more specific
      claim about *why*. A named-misconception taxonomy beyond this is
      future tutoring-policy work, not something to approximate here.

    Returns `None` for `"equivalent"` (nothing to diagnose) and for
    `"unsupported"`/`"indeterminate"` (nothing was confirmed wrong)."""
    if transition == "narrowing":
        return "lost_solution_branch"
    if transition == "broadening":
        return "introduced_extraneous_root"
    if transition in ("incompatible", "partial_overlap"):
        if isinstance(old_set, sympy.FiniteSet) and isinstance(new_set, sympy.FiniteSet):
            negated_old = sympy.FiniteSet(*(-element for element in old_set.args))
            if negated_old == new_set:
                return "sign_error"
        return "general_error"
    return None


_CONCEPTUAL_HINT: dict[Category, str] = {
    "linear": (
        "This is a linear equation — it can be solved by isolating the "
        "variable using inverse operations on both sides."
    ),
    "quadratic": (
        "This is a quadratic equation — consider factoring, completing "
        "the square, or the quadratic formula."
    ),
    "unsupported": "This problem is outside the currently supported verification scope.",
}

_DIRECTIONAL_HINT: dict[Category, str] = {
    "linear": (
        "Combine like terms on each side first, then apply the same "
        "operation to both sides to isolate the variable."
    ),
    "quadratic": (
        "Try moving every term to one side so the equation equals zero, "
        "then look for a way to factor it or apply the quadratic formula."
    ),
    "unsupported": "No further guidance is available for this problem.",
}


def generate_hint_ladder(category: Category, true_set: sympy.Set) -> list[str]:
    """Four hint levels, least to most revealing. Only level 4 names the
    actual solution(s) — callers are responsible for not returning it
    unless the student explicitly asked for that level (see
    `docs/jee-solving.md`'s tutoring-policy section); this function
    itself always computes all four, it does not gate disclosure.
    Levels 1–2 are fixed guidance text per category; levels 3–4 are
    derived from the already-computed, verified `true_set` — never
    invented."""
    level1 = _CONCEPTUAL_HINT[category]
    level2 = _DIRECTIONAL_HINT[category]

    if true_set == sympy.S.EmptySet:
        level3 = "This equation has no real solutions."
        level4 = "There is no real solution — the real solution set is empty."
    elif true_set == sympy.S.Reals:
        level3 = "This equation holds for every real number — it's an identity."
        level4 = "Every real number is a solution."
    elif isinstance(true_set, sympy.FiniteSet):
        count = len(true_set)
        level3 = f"This equation has exactly {count} real solution(s)."
        values = ", ".join(sorted(str(v) for v in true_set.args))
        level4 = f"The real solution(s): {values}."
    else:
        level3 = "The exact solution structure could not be determined."
        level4 = "The exact solution could not be determined."

    return [level1, level2, level3, level4]


def hints_for_equation(equation_text: str, variable: str, hint_level: int) -> list[str] | None:
    """Convenience wrapper: parses and classifies `equation_text`, then
    returns `generate_hint_ladder(...)` sliced to `hint_level` (1–4).
    `None` if the equation itself can't be classified/solved — there is
    no verified structure to hint from in that case."""
    try:
        lhs, rhs = parse_equation(equation_text, variable)
    except ParseError:
        return None
    var = sympy.Symbol(variable)
    category = classify(lhs, rhs, var)
    if category is None:
        return None
    true_set = solve_real(lhs, rhs, var)
    if not is_decidable_set(true_set):
        return None
    return generate_hint_ladder(category, true_set)[:hint_level]


def check_steps(
    equation_text: str,
    variable: str,
    step_texts: list[str],
    solution_texts: list[str] | None,
) -> StepCheckOutcome:
    """Verifies an ordered chain of student equation steps against the
    original equation (step 0), and — independently — the proposed final
    `solution_texts` against the *original* equation via `verify()`
    (never against whatever the student's last step happened to be).
    `solution_texts=None` means no final answer was submitted; `[]` is
    the existing Phase 2A claim of "no real solutions" (see `verify()`).

    A step whose own equation can't be parsed or classified, or whose
    solution set can't be decided, breaks verification continuity: it is
    reported honestly as `"unsupported"`/`"indeterminate"` rather than
    silently reusing stale state, and the *next* step (if its own
    equation checks out) resumes comparisons from its own solution set —
    the gap itself is marked, not papered over, but one bad step doesn't
    permanently poison every step after it.
    """

    def _no_checkable_result(detail: str) -> StepCheckOutcome:
        return StepCheckOutcome(
            status="unsupported",
            steps=[],
            first_invalid_step_index=None,
            first_non_equivalent_step_index=None,
            chain_preserves_solution_set=None,
            deterministic=True,
            detail=detail,
            final_answer=None,
        )

    try:
        lhs0, rhs0 = parse_equation(equation_text, variable)
    except ParseError as exc:
        return _no_checkable_result(f"could not parse the original equation: {exc}")

    var = sympy.Symbol(variable)
    category0 = classify(lhs0, rhs0, var)
    if category0 is None:
        _, detail0 = _classification_failure(lhs0, rhs0, var)
        return _no_checkable_result(detail0)

    original_set = solve_real(lhs0, rhs0, var)
    previous_set: sympy.Set | None = original_set if is_decidable_set(original_set) else None

    steps: list[StepOutcome] = []
    first_invalid: int | None = None
    first_non_equivalent: int | None = None
    continuity_lost = previous_set is None
    any_indeterminate = False

    for index, step_text in enumerate(step_texts, start=1):
        try:
            lhs_i, rhs_i = parse_equation(step_text, variable)
        except ParseError as exc:
            steps.append(
                StepOutcome(
                    index,
                    step_text,
                    "unsupported",
                    "unsupported",
                    f"could not parse this step: {exc}",
                )
            )
            previous_set = None
            continuity_lost = True
            continue

        category_i = classify(lhs_i, rhs_i, var)
        if category_i is None:
            _, detail_i = _classification_failure(lhs_i, rhs_i, var)
            steps.append(StepOutcome(index, step_text, "unsupported", "unsupported", detail_i))
            previous_set = None
            continuity_lost = True
            continue

        current_set = solve_real(lhs_i, rhs_i, var)
        if not is_decidable_set(current_set):
            steps.append(
                StepOutcome(
                    index,
                    step_text,
                    category_i,
                    "indeterminate",
                    "the exact real solution set for this step could not be established",
                )
            )
            previous_set = None
            continuity_lost = True
            any_indeterminate = True
            continue

        if previous_set is None:
            steps.append(
                StepOutcome(
                    index,
                    step_text,
                    category_i,
                    "unsupported",
                    "verification continuity was lost at an earlier step",
                )
            )
            previous_set = current_set  # recover: resume comparisons from here
            continue

        transition = classify_transition(previous_set, current_set)
        misconception = classify_misconception(previous_set, current_set, transition)
        steps.append(
            StepOutcome(
                index,
                step_text,
                category_i,
                transition,
                _TRANSITION_DETAIL[transition],
                misconception,
            )
        )
        previous_set = current_set

        if transition != "equivalent" and first_non_equivalent is None:
            first_non_equivalent = index
        # "incompatible" and "partial_overlap" both lose at least one
        # genuine solution without a legitimate subset relationship —
        # that loss is provably wrong regardless of what else the step
        # also introduced, so both set first_invalid (unlike pure
        # "narrowing"/"broadening", which never lose without cause).
        if transition in ("incompatible", "partial_overlap") and first_invalid is None:
            first_invalid = index

    if first_invalid is not None:
        status: StepCheckStatus = "invalid_step_found"
    elif continuity_lost:
        status = "verification_incomplete"
    elif first_non_equivalent is not None:
        status = "valid_but_incomplete"
    else:
        status = "all_steps_equivalent"

    chain_preserves_solution_set: bool | None
    if first_non_equivalent is not None:
        chain_preserves_solution_set = False
    elif continuity_lost:
        chain_preserves_solution_set = None
    else:
        chain_preserves_solution_set = True

    final_answer = (
        None if solution_texts is None else verify(equation_text, variable, solution_texts)
    )

    return StepCheckOutcome(
        status=status,
        steps=steps,
        first_invalid_step_index=first_invalid,
        first_non_equivalent_step_index=first_non_equivalent,
        chain_preserves_solution_set=chain_preserves_solution_set,
        deterministic=not any_indeterminate,
        detail=_STATUS_DETAIL[status],
        final_answer=final_answer,
    )


# --- Phase 3: calculus verification (derivatives / indefinite integrals) --

CalculusStatus = Literal[
    "computed", "verified_correct", "verified_incorrect", "unsupported", "indeterminate"
]


@dataclass
class CalculusOutcome:
    """`computed` is this engine's own closed-form result (via SymPy
    `diff`/`integrate`), shown for reference only — it is a *computed*
    answer, not by itself a verdict on the student's submission.
    `status` is always reached by independently checking the *student's*
    proposed answer (`verified`), never by comparing the two for a
    textual/structural match — two different closed forms can be
    equally correct (e.g. differing by a constant or a trig identity).
    `deterministic` is `False` only for `"indeterminate"`, exactly the
    convention `VerificationOutcome` already uses."""

    status: CalculusStatus
    operation: Literal["derivative", "integral"]
    computed: str | None
    deterministic: bool
    reason: str | None
    detail: str


def compute_derivative(expression_text: str, variable: str) -> CalculusOutcome:
    """Computes (does not verify anything — there is no proposed answer
    to check) the exact derivative of `expression_text`. Used by the
    deterministic problem router (`axiom/problem_router.py`) to answer
    an explicit computation command directly."""
    var = sympy.Symbol(variable)
    try:
        expr = parse_calculus_expression(expression_text, variable)
    except ParseError as exc:
        return CalculusOutcome(
            status="unsupported",
            operation="derivative",
            computed=None,
            deterministic=True,
            reason="unparseable_expression",
            detail=f"could not parse the expression: {exc}",
        )
    computed = sympy.diff(expr, var)
    return CalculusOutcome(
        status="computed",
        operation="derivative",
        computed=str(computed),
        deterministic=True,
        reason=None,
        detail="exact derivative computed",
    )


def compute_integral(expression_text: str, variable: str) -> CalculusOutcome:
    """Computes the exact indefinite integral of `expression_text` (no
    proposed answer to verify). See `compute_derivative`."""
    var = sympy.Symbol(variable)
    try:
        expr = parse_calculus_expression(expression_text, variable)
    except ParseError as exc:
        return CalculusOutcome(
            status="unsupported",
            operation="integral",
            computed=None,
            deterministic=True,
            reason="unparseable_expression",
            detail=f"could not parse the expression: {exc}",
        )
    result = sympy.integrate(expr, var)
    if isinstance(result, sympy.Integral):
        return CalculusOutcome(
            status="unsupported",
            operation="integral",
            computed=None,
            deterministic=True,
            reason="no_closed_form",
            detail="no closed-form antiderivative was found",
        )
    return CalculusOutcome(
        status="computed",
        operation="integral",
        computed=str(result),
        deterministic=True,
        reason=None,
        detail="exact antiderivative computed",
    )


def check_derivative(expression_text: str, variable: str, proposed_text: str) -> CalculusOutcome:
    """Verifies a proposed derivative of `expression_text` by exact
    symbolic equality against SymPy's own `diff()` — not by comparing
    the student's text to a reference string, which would reject
    equally valid alternate forms."""
    var = sympy.Symbol(variable)
    try:
        expr = parse_calculus_expression(expression_text, variable)
    except ParseError as exc:
        return CalculusOutcome(
            status="unsupported",
            operation="derivative",
            computed=None,
            deterministic=True,
            reason="unparseable_expression",
            detail=f"could not parse the expression: {exc}",
        )
    try:
        proposed = parse_calculus_expression(proposed_text, variable)
    except ParseError as exc:
        return CalculusOutcome(
            status="unsupported",
            operation="derivative",
            computed=None,
            deterministic=True,
            reason="unparseable_proposed_derivative",
            detail=f"could not parse the proposed derivative: {exc}",
        )

    computed = sympy.diff(expr, var)
    decision = (proposed - computed).equals(0)
    if decision is None:
        return CalculusOutcome(
            status="indeterminate",
            operation="derivative",
            computed=str(computed),
            deterministic=False,
            reason="equality_undecidable",
            detail="could not exactly decide whether the proposed derivative is correct",
        )
    if decision:
        return CalculusOutcome(
            status="verified_correct",
            operation="derivative",
            computed=str(computed),
            deterministic=True,
            reason=None,
            detail="the proposed derivative is exactly correct",
        )
    return CalculusOutcome(
        status="verified_incorrect",
        operation="derivative",
        computed=str(computed),
        deterministic=True,
        reason="incorrect_derivative",
        detail="the proposed derivative does not match the exact derivative",
    )


def check_integral(expression_text: str, variable: str, proposed_text: str) -> CalculusOutcome:
    """Verifies a proposed *indefinite* integral by differentiating the
    student's own candidate back and checking it exactly equals the
    original expression — the correct general method, and one that
    automatically handles an arbitrary additive constant (a derivative
    removes it) without special-casing `+ C`. Never checks the
    candidate against SymPy's own `integrate()` result directly: two
    antiderivatives can be equally valid while looking algebraically
    different (e.g. differing by a constant from a trig identity).

    Only indefinite integrals are supported. Definite integrals (bounds,
    singularities) are explicitly out of scope for this milestone — see
    `docs/verification.md`."""
    var = sympy.Symbol(variable)
    try:
        expr = parse_calculus_expression(expression_text, variable)
    except ParseError as exc:
        return CalculusOutcome(
            status="unsupported",
            operation="integral",
            computed=None,
            deterministic=True,
            reason="unparseable_expression",
            detail=f"could not parse the expression: {exc}",
        )
    try:
        proposed = parse_calculus_expression(proposed_text, variable)
    except ParseError as exc:
        return CalculusOutcome(
            status="unsupported",
            operation="integral",
            computed=None,
            deterministic=True,
            reason="unparseable_proposed_antiderivative",
            detail=f"could not parse the proposed antiderivative: {exc}",
        )

    computed_antiderivative = sympy.integrate(expr, var)
    computed_str = (
        None
        if isinstance(computed_antiderivative, sympy.Integral)
        else str(computed_antiderivative)
    )

    decision = (sympy.diff(proposed, var) - expr).equals(0)
    if decision is None:
        return CalculusOutcome(
            status="indeterminate",
            operation="integral",
            computed=computed_str,
            deterministic=False,
            reason="equality_undecidable",
            detail="could not exactly decide whether the proposed antiderivative is correct",
        )
    if decision:
        return CalculusOutcome(
            status="verified_correct",
            operation="integral",
            computed=computed_str,
            deterministic=True,
            reason=None,
            detail="differentiating the proposed antiderivative exactly reproduces the original",
        )
    return CalculusOutcome(
        status="verified_incorrect",
        operation="integral",
        computed=computed_str,
        deterministic=True,
        reason="incorrect_antiderivative",
        detail="differentiating the proposed antiderivative does not reproduce the original",
    )


# --- Phase 3: combinatorics / probability (exact rational) ----------------

CombinatoricsOperation = Literal["ncr", "npr", "probability"]
CombinatoricsStatus = Literal[
    "computed", "verified_correct", "verified_incorrect", "unsupported", "indeterminate"
]

MAX_COMBINATORICS_N = 100


@dataclass
class CombinatoricsOutcome:
    """`status` is `"computed"` when no `proposed_answer` was given —
    `computed` is then the only result, with nothing to verify.
    `"verified_correct"`/`"verified_incorrect"` only apply once a
    proposed answer was actually checked against the exact value."""

    status: CombinatoricsStatus
    operation: CombinatoricsOperation
    computed: str | None
    deterministic: bool
    reason: str | None
    detail: str


class _CombinatoricsInputError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def _compute_combinatorics(
    operation: CombinatoricsOperation, n: int, r: int, favorable: int | None
) -> sympy.Rational | sympy.Integer:
    """Exact rational/integer arithmetic only — `sympy.binomial`,
    `sympy.factorial` — never a floating-point approximation."""
    if r > n:
        raise _CombinatoricsInputError("r_exceeds_n", "r cannot exceed n")

    if operation == "ncr":
        return sympy.binomial(n, r)
    if operation == "npr":
        return sympy.factorial(n) / sympy.factorial(n - r)

    if favorable is None:
        raise _CombinatoricsInputError(
            "favorable_required", "'favorable' is required for a probability calculation"
        )
    if favorable > n:
        raise _CombinatoricsInputError("favorable_exceeds_n", "favorable cannot exceed n")
    total_ways = sympy.binomial(n, r)
    if total_ways == 0:
        raise _CombinatoricsInputError(
            "no_valid_selection", "there is no way to choose r items from n"
        )
    favorable_ways = sympy.binomial(favorable, r)
    return sympy.Rational(favorable_ways, total_ways)


def check_combinatorics(
    operation: CombinatoricsOperation,
    n: int,
    r: int,
    favorable: int | None = None,
    proposed_answer: str | None = None,
) -> CombinatoricsOutcome:
    """`"ncr"`/`"npr"` are plain exact counts. `"probability"` is
    deliberately narrow and explicit, not a general solver: it is the
    probability that an r-item selection drawn from n items (all
    equally likely, no replacement) consists entirely of items from a
    `favorable` subset of size `favorable` — the classic "all drawn
    balls are red" JEE pattern — computed as
    `C(favorable, r) / C(n, r)`. Any other probability question (e.g.
    "at least one", conditional probability, non-uniform outcomes) is
    out of scope and not something this function attempts to recognize
    or guess at."""
    try:
        computed = _compute_combinatorics(operation, n, r, favorable)
    except _CombinatoricsInputError as exc:
        return CombinatoricsOutcome(
            status="unsupported",
            operation=operation,
            computed=None,
            deterministic=True,
            reason=exc.reason,
            detail=str(exc),
        )

    if proposed_answer is None:
        return CombinatoricsOutcome(
            status="computed",
            operation=operation,
            computed=str(computed),
            deterministic=True,
            reason=None,
            detail="exact value computed",
        )

    try:
        proposed = parse_candidate(proposed_answer)
    except ParseError as exc:
        return CombinatoricsOutcome(
            status="unsupported",
            operation=operation,
            computed=str(computed),
            deterministic=True,
            reason="unparseable_proposed_answer",
            detail=f"could not parse the proposed answer: {exc}",
        )

    decision = (proposed - computed).equals(0)
    if decision is None:
        return CombinatoricsOutcome(
            status="indeterminate",
            operation=operation,
            computed=str(computed),
            deterministic=False,
            reason="equality_undecidable",
            detail="could not exactly decide whether the proposed answer is correct",
        )
    if decision:
        return CombinatoricsOutcome(
            status="verified_correct",
            operation=operation,
            computed=str(computed),
            deterministic=True,
            reason=None,
            detail="the proposed answer is exactly correct",
        )
    return CombinatoricsOutcome(
        status="verified_incorrect",
        operation=operation,
        computed=str(computed),
        deterministic=True,
        reason="incorrect_value",
        detail="the proposed answer does not match the exact value",
    )
