"""Bounded practice-problem generation (Phase 3).

Two skills, each with explicit, testable difficulty definitions:

- `"linear_equations"` / `"easy"` — `a*x + b = c`, small integer
  coefficients, exactly one integer solution.
- `"quadratic_equations"` / `"medium"` — `(x-p)*(x-q) = 0` expanded,
  two small integer roots.
- `"quadratic_equations"` / `"hard"` — `x^2 - k = 0` for a small
  non-perfect-square `k`, two irrational (`sqrt`) roots.

Every generated problem is re-parsed and re-solved through the exact
same `axiom/verification/engine.py` functions `/api/v1/verify` and
`/api/v1/check-work` use — never shown unless that independent check
confirms it classifies as expected and has a decidable solution set.
This is deliberately a small, hand-designed template bank, not an LLM
generator — a generated problem is exact algebra, not a plausible-
looking guess.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import sympy

from axiom.verification.engine import Category, classify, is_decidable_set, solve_real
from axiom.verification.parser import ParseError, parse_equation

Skill = Literal["linear_equations", "quadratic_equations"]
Difficulty = Literal["easy", "medium", "hard"]

MAX_GENERATION_ATTEMPTS = 10


@dataclass
class PracticeProblem:
    skill: Skill
    difficulty: Difficulty
    equation: str
    variable: str


def _generate_easy_linear() -> str:
    coefficient = random.choice([1, 2, 3, 4, 5])
    true_x = random.randint(-10, 10)
    constant = random.randint(-10, 10)
    result = coefficient * true_x + constant
    lhs = f"{coefficient}*x"
    if constant != 0:
        lhs += f" + {constant}" if constant > 0 else f" - {-constant}"
    return f"{lhs} = {result}"


def _generate_medium_quadratic() -> str:
    root_a = random.randint(-10, 10)
    root_b = random.randint(-10, 10)
    linear_coeff = -(root_a + root_b)
    constant = root_a * root_b
    terms = ["x^2"]
    if linear_coeff != 0:
        terms.append(f"+ {linear_coeff}*x" if linear_coeff > 0 else f"- {-linear_coeff}*x")
    if constant != 0:
        terms.append(f"+ {constant}" if constant > 0 else f"- {-constant}")
    return " ".join(terms) + " = 0"


def _generate_hard_quadratic() -> str:
    while True:
        k = random.randint(2, 30)
        root = math.isqrt(k)
        if root * root != k:
            return f"x^2 - {k} = 0"


_GENERATORS: dict[tuple[Skill, Difficulty], Callable[[], str]] = {
    ("linear_equations", "easy"): _generate_easy_linear,
    ("quadratic_equations", "medium"): _generate_medium_quadratic,
    ("quadratic_equations", "hard"): _generate_hard_quadratic,
}

_EXPECTED_CATEGORY: dict[Skill, Category] = {
    "linear_equations": "linear",
    "quadratic_equations": "quadratic",
}


def generate_practice_problem(skill: Skill, difficulty: Difficulty) -> PracticeProblem | None:
    """`None` only if `MAX_GENERATION_ATTEMPTS` independent verification
    checks all failed — this should not happen in practice given the
    templates above are constructed to always produce a classifiable,
    decidable equation, but a generated problem is never shown without
    this check actually passing."""
    generator = _GENERATORS.get((skill, difficulty))
    if generator is None:
        return None

    expected_category = _EXPECTED_CATEGORY[skill]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        equation_text = generator()
        if _verify_problem_is_valid(equation_text, "x", expected_category):
            return PracticeProblem(
                skill=skill, difficulty=difficulty, equation=equation_text, variable="x"
            )
    return None


def _verify_problem_is_valid(
    equation_text: str, variable: str, expected_category: Category
) -> bool:
    try:
        lhs, rhs = parse_equation(equation_text, variable)
    except ParseError:
        return False
    var = sympy.Symbol(variable)
    category = classify(lhs, rhs, var)
    if category != expected_category:
        return False
    true_set = solve_real(lhs, rhs, var)
    return is_decidable_set(true_set)


def select_next_problem(
    linear_mastery: float, quadratic_mastery: float
) -> tuple[Skill, Difficulty]:
    """A simple, explainable, deterministic rule: practice the weaker
    skill; within quadratics, escalate difficulty once mastery clears
    0.6. Not adaptive in a statistically validated sense — a documented
    heuristic, not a claimed learning-science result."""
    if linear_mastery < 0.6:
        return "linear_equations", "easy"
    if quadratic_mastery < 0.6:
        return "quadratic_equations", "medium"
    return "quadratic_equations", "hard"
