import re

import sympy

from axiom.practice import (
    MAX_GENERATION_ATTEMPTS,
    generate_practice_problem,
    select_next_problem,
)
from axiom.verification.engine import classify, is_decidable_set, solve_real
from axiom.verification.parser import parse_equation

N_SAMPLES = 30


class TestGenerateEasyLinear:
    def test_generates_a_classified_linear_equation(self) -> None:
        for _ in range(N_SAMPLES):
            problem = generate_practice_problem("linear_equations", "easy")
            assert problem is not None
            assert problem.skill == "linear_equations"
            assert problem.difficulty == "easy"
            lhs, rhs = parse_equation(problem.equation, problem.variable)
            var = sympy.Symbol(problem.variable)
            assert classify(lhs, rhs, var) == "linear"

    def test_has_exactly_one_decidable_real_solution(self) -> None:
        for _ in range(N_SAMPLES):
            problem = generate_practice_problem("linear_equations", "easy")
            assert problem is not None
            lhs, rhs = parse_equation(problem.equation, problem.variable)
            var = sympy.Symbol(problem.variable)
            true_set = solve_real(lhs, rhs, var)
            assert is_decidable_set(true_set)
            assert isinstance(true_set, sympy.FiniteSet)
            assert len(true_set) == 1

    def test_uses_only_the_restricted_grammar(self) -> None:
        for _ in range(N_SAMPLES):
            problem = generate_practice_problem("linear_equations", "easy")
            assert problem is not None
            assert re.fullmatch(r"[0-9x+\-*= ]+", problem.equation)


class TestGenerateMediumQuadratic:
    def test_generates_a_classified_quadratic_with_two_or_one_real_roots(self) -> None:
        for _ in range(N_SAMPLES):
            problem = generate_practice_problem("quadratic_equations", "medium")
            assert problem is not None
            lhs, rhs = parse_equation(problem.equation, problem.variable)
            var = sympy.Symbol(problem.variable)
            assert classify(lhs, rhs, var) == "quadratic"
            true_set = solve_real(lhs, rhs, var)
            assert is_decidable_set(true_set)
            assert isinstance(true_set, sympy.FiniteSet)
            assert all(not root.free_symbols and root.is_rational for root in true_set.args)


class TestGenerateHardQuadratic:
    def test_generates_irrational_roots(self) -> None:
        for _ in range(N_SAMPLES):
            problem = generate_practice_problem("quadratic_equations", "hard")
            assert problem is not None
            lhs, rhs = parse_equation(problem.equation, problem.variable)
            var = sympy.Symbol(problem.variable)
            true_set = solve_real(lhs, rhs, var)
            assert is_decidable_set(true_set)
            assert isinstance(true_set, sympy.FiniteSet)
            assert any(not root.is_rational for root in true_set.args)


class TestGenerationBounds:
    def test_unknown_skill_difficulty_pair_returns_none(self) -> None:
        assert generate_practice_problem("linear_equations", "hard") is None

    def test_max_attempts_is_a_small_bounded_constant(self) -> None:
        assert 1 <= MAX_GENERATION_ATTEMPTS <= 50


class TestSelectNextProblem:
    def test_weak_linear_mastery_selects_easy_linear(self) -> None:
        assert select_next_problem(0.1, 0.5) == ("linear_equations", "easy")

    def test_strong_linear_weak_quadratic_selects_medium_quadratic(self) -> None:
        assert select_next_problem(0.9, 0.2) == ("quadratic_equations", "medium")

    def test_strong_both_selects_hard_quadratic(self) -> None:
        assert select_next_problem(0.9, 0.9) == ("quadratic_equations", "hard")

    def test_boundary_at_0_6_is_not_treated_as_weak(self) -> None:
        # Exactly 0.6 fails the strict "< 0.6" weak-skill check on both
        # sides, so the rule falls through to the hardest tier.
        assert select_next_problem(0.6, 0.6) == ("quadratic_equations", "hard")
