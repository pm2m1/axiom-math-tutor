import sympy

from axiom.verification.engine import (
    check_steps,
    classify_misconception,
    generate_hint_ladder,
    hints_for_equation,
)


class TestClassifyMisconception:
    def test_narrowing_is_lost_solution_branch(self) -> None:
        old = sympy.FiniteSet(-1, 1)
        new = sympy.FiniteSet(1)
        assert classify_misconception(old, new, "narrowing") == "lost_solution_branch"

    def test_broadening_is_introduced_extraneous_root(self) -> None:
        old = sympy.FiniteSet(1)
        new = sympy.FiniteSet(-1, 1)
        assert classify_misconception(old, new, "broadening") == "introduced_extraneous_root"

    def test_incompatible_sign_flip_is_sign_error(self) -> None:
        old = sympy.FiniteSet(4)
        new = sympy.FiniteSet(-4)
        assert classify_misconception(old, new, "incompatible") == "sign_error"

    def test_incompatible_sign_flip_multi_element(self) -> None:
        old = sympy.FiniteSet(2, 3)
        new = sympy.FiniteSet(-2, -3)
        assert classify_misconception(old, new, "incompatible") == "sign_error"

    def test_incompatible_unrelated_values_is_general_error(self) -> None:
        old = sympy.FiniteSet(4)
        new = sympy.FiniteSet(5)
        assert classify_misconception(old, new, "incompatible") == "general_error"

    def test_partial_overlap_is_general_error(self) -> None:
        old = sympy.FiniteSet(1, 2)
        new = sympy.FiniteSet(2, 3)
        assert classify_misconception(old, new, "partial_overlap") == "general_error"

    def test_equivalent_has_no_misconception(self) -> None:
        old = sympy.FiniteSet(1, 2)
        assert classify_misconception(old, old, "equivalent") is None

    def test_sign_flip_does_not_apply_to_non_finite_sets(self) -> None:
        # S.Reals/S.EmptySet aren't FiniteSet, so the sign check can't
        # run — must fall back to general_error, not raise.
        result = classify_misconception(sympy.S.Reals, sympy.S.EmptySet, "incompatible")
        assert result == "general_error"


class TestCheckStepsSetsMisconception:
    def test_narrowing_step_has_misconception(self) -> None:
        outcome = check_steps("x^2=1", "x", ["x=1"], None)
        assert outcome.steps[0].misconception == "lost_solution_branch"

    def test_sign_error_step_detected_through_check_steps(self) -> None:
        outcome = check_steps("2*x=8", "x", ["x=-4"], None)
        assert outcome.steps[0].transition == "incompatible"
        assert outcome.steps[0].misconception == "sign_error"

    def test_equivalent_step_has_no_misconception(self) -> None:
        outcome = check_steps("2*x=8", "x", ["x=4"], None)
        assert outcome.steps[0].misconception is None

    def test_unsupported_step_has_no_misconception(self) -> None:
        outcome = check_steps("2*x=8", "x", ["x + y = 1"], None)
        assert outcome.steps[0].misconception is None


class TestGenerateHintLadder:
    def test_returns_four_levels(self) -> None:
        hints = generate_hint_ladder("linear", sympy.FiniteSet(3))
        assert len(hints) == 4

    def test_level_four_names_the_solution(self) -> None:
        hints = generate_hint_ladder("linear", sympy.FiniteSet(3))
        assert "3" in hints[3]

    def test_level_one_through_three_do_not_name_the_solution(self) -> None:
        hints = generate_hint_ladder("linear", sympy.FiniteSet(3))
        for hint in hints[:3]:
            assert "3." not in hint and hint != hints[3]

    def test_no_real_solutions_case(self) -> None:
        hints = generate_hint_ladder("quadratic", sympy.S.EmptySet)
        assert "no real solutions" in hints[2]

    def test_identity_case(self) -> None:
        hints = generate_hint_ladder("linear", sympy.S.Reals)
        assert "every real number" in hints[2]

    def test_irrational_solutions_named_at_level_four(self) -> None:
        hints = generate_hint_ladder("quadratic", sympy.FiniteSet(sympy.sqrt(2), -sympy.sqrt(2)))
        assert "sqrt(2)" in hints[3]


class TestHintsForEquation:
    def test_slices_to_requested_level(self) -> None:
        assert len(hints_for_equation("2*x=8", "x", 1)) == 1
        assert len(hints_for_equation("2*x=8", "x", 3)) == 3
        assert len(hints_for_equation("2*x=8", "x", 4)) == 4

    def test_level_four_only_when_explicitly_requested(self) -> None:
        hints_level_3 = hints_for_equation("2*x=8", "x", 3)
        assert all("4" not in h for h in hints_level_3)  # the solution (4) is not named yet
        hints_level_4 = hints_for_equation("2*x=8", "x", 4)
        assert "4" in hints_level_4[3]

    def test_unsupported_equation_returns_none(self) -> None:
        assert hints_for_equation("x^3=8", "x", 1) is None

    def test_malformed_equation_returns_none(self) -> None:
        assert hints_for_equation("not an equation", "x", 1) is None
