"""Deterministic SymPy verification engine (Phase 2A).

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
`verify()` is only the orchestration of those pieces.
"""

from dataclasses import dataclass, field
from typing import Literal

import sympy
from sympy.polys.polyerrors import PolynomialError

from axiom.verification.parser import ParseError, parse_candidate, parse_equation

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
