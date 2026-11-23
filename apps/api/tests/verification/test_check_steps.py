import sympy

from axiom.verification.engine import check_steps, classify_transition

X = sympy.Symbol("x")


class TestClassifyTransition:
    def test_equal_sets_are_equivalent(self) -> None:
        assert classify_transition(sympy.FiniteSet(1, 2), sympy.FiniteSet(1, 2)) == "equivalent"

    def test_both_empty_is_equivalent(self) -> None:
        assert classify_transition(sympy.S.EmptySet, sympy.S.EmptySet) == "equivalent"

    def test_both_reals_is_equivalent(self) -> None:
        assert classify_transition(sympy.S.Reals, sympy.S.Reals) == "equivalent"

    def test_proper_nonempty_subset_is_narrowing(self) -> None:
        assert classify_transition(sympy.FiniteSet(-1, 1), sympy.FiniteSet(1)) == "narrowing"

    def test_reals_down_to_finite_is_narrowing(self) -> None:
        assert classify_transition(sympy.S.Reals, sympy.FiniteSet(1, 2)) == "narrowing"

    def test_proper_superset_is_broadening(self) -> None:
        assert classify_transition(sympy.FiniteSet(1), sympy.FiniteSet(-1, 1)) == "broadening"

    def test_empty_to_nonempty_is_broadening(self) -> None:
        assert classify_transition(sympy.S.EmptySet, sympy.FiniteSet(1)) == "broadening"

    def test_finite_up_to_reals_is_broadening(self) -> None:
        assert classify_transition(sympy.FiniteSet(1, 2), sympy.S.Reals) == "broadening"

    def test_disjoint_nonempty_sets_are_incompatible(self) -> None:
        assert classify_transition(sympy.FiniteSet(4), sympy.FiniteSet(5)) == "incompatible"

    def test_losing_everything_is_incompatible(self) -> None:
        assert classify_transition(sympy.FiniteSet(1, 2), sympy.S.EmptySet) == "incompatible"

    def test_reals_to_empty_is_incompatible(self) -> None:
        assert classify_transition(sympy.S.Reals, sympy.S.EmptySet) == "incompatible"

    def test_partial_overlap_is_its_own_category(self) -> None:
        old = sympy.FiniteSet(1, 2)
        new = sympy.FiniteSet(2, 3)
        assert classify_transition(old, new) == "partial_overlap"


def _run(equation: str, steps: list[str], solutions: list[str] | None = None):
    return check_steps(equation, "x", steps, solutions)


class TestValidTransformations:
    def test_two_valid_linear_steps(self) -> None:
        outcome = _run("2*x+3=11", ["2*x=8", "x=4"])
        assert outcome.status == "all_steps_equivalent"
        assert outcome.chain_preserves_solution_set is True
        assert outcome.first_invalid_step_index is None
        assert all(s.transition == "equivalent" for s in outcome.steps)

    def test_single_valid_linear_step(self) -> None:
        outcome = _run("3*x=9", ["x=3"])
        assert outcome.status == "all_steps_equivalent"

    def test_quadratic_factoring_is_equivalent(self) -> None:
        outcome = _run("x^2-5*x+6=0", ["(x-2)*(x-3)=0"])
        assert outcome.status == "all_steps_equivalent"
        assert outcome.steps[0].category == "quadratic"

    def test_rearranged_terms_between_sides_is_equivalent(self) -> None:
        outcome = _run("2*x+3=11", ["2*x-11=-3"])
        assert outcome.status == "all_steps_equivalent"


class TestFirstIncorrectStep:
    def test_second_step_is_wrong(self) -> None:
        outcome = _run("2*x+3=11", ["2*x=8", "x=5"])
        assert outcome.status == "invalid_step_found"
        assert outcome.first_invalid_step_index == 2
        assert outcome.steps[0].transition == "equivalent"
        assert outcome.steps[1].transition == "incompatible"

    def test_single_step_is_wrong(self) -> None:
        outcome = _run("3*x=9", ["x=4"])
        assert outcome.first_invalid_step_index == 1

    def test_several_valid_steps_then_one_wrong(self) -> None:
        outcome = _run("x^2-5*x+6=0", ["(x-2)*(x-3)=0", "x-2=0", "x=5"])
        assert outcome.status == "invalid_step_found"
        assert outcome.first_invalid_step_index == 3
        assert outcome.steps[0].transition == "equivalent"
        assert outcome.steps[1].transition == "narrowing"
        assert outcome.steps[2].transition == "incompatible"

    def test_wrong_first_step_then_apparently_valid_algebra(self) -> None:
        # x=5 is wrong immediately; doubling both sides afterward is a
        # valid *operation* but on an already-incorrect equation.
        outcome = _run("2*x+3=11", ["x=5", "2*x=10"])
        assert outcome.status == "invalid_step_found"
        assert outcome.first_invalid_step_index == 1
        assert outcome.steps[1].transition == "equivalent"


class TestNonEquivalentTransformations:
    def test_narrowing_from_quadratic_to_one_root(self) -> None:
        outcome = _run("x^2=1", ["x=1"])
        assert outcome.status == "valid_but_incomplete"
        assert outcome.steps[0].transition == "narrowing"
        assert outcome.first_invalid_step_index is None
        assert outcome.first_non_equivalent_step_index == 1
        assert outcome.chain_preserves_solution_set is False

    def test_broadening_from_one_root_to_quadratic(self) -> None:
        outcome = _run("x=1", ["x^2=1"])
        assert outcome.status == "valid_but_incomplete"
        assert outcome.steps[0].transition == "broadening"

    def test_repeated_root_stays_equivalent(self) -> None:
        outcome = _run("x^2-2*x+1=0", ["(x-1)^2=0", "x=1"])
        assert outcome.status == "all_steps_equivalent"

    def test_no_real_solution_equation_preserved(self) -> None:
        outcome = _run("x^2+1=0", ["x^2=-1"])
        assert outcome.status == "all_steps_equivalent"

    def test_identity_step_without_the_variable_is_unsupported(self) -> None:
        # "2*x = 2*x" -> "0 = 0" is a common simplification, but once the
        # variable disappears entirely the step itself can't be classified
        # (same rule Phase 2A applies to the original equation).
        outcome = _run("2*x=2*x", ["0=0"])
        assert outcome.steps[0].transition == "unsupported"
        assert outcome.status == "verification_incomplete"

    def test_contradiction_step_without_the_variable_is_unsupported(self) -> None:
        outcome = _run("3*x=3*x+1", ["0=1"])
        assert outcome.steps[0].transition == "unsupported"

    def test_overlapping_non_equivalent_sets_is_provably_invalid(self) -> None:
        # {1,2} -> {2,3}: loses a genuine solution (1) while introducing an
        # unverified one (3) — not a legitimate branch (pure narrowing
        # never adds) or extension (pure broadening never drops), so this
        # must be flagged as the first provably invalid step, not merely
        # "incomplete" alongside narrowing/broadening.
        outcome = _run("(x-1)*(x-2)=0", ["(x-2)*(x-3)=0"])
        assert outcome.steps[0].transition == "partial_overlap"
        assert outcome.status == "invalid_step_found"
        assert outcome.first_invalid_step_index == 1
        assert outcome.first_non_equivalent_step_index == 1
        assert outcome.chain_preserves_solution_set is False

    def test_partial_overlap_after_valid_steps_is_the_first_error(self) -> None:
        outcome = _run(
            "(x-1)*(x-2)=0",
            ["(x-1)*(x-2)=0", "(x-2)*(x-3)=0", "x=3"],
        )
        assert outcome.first_invalid_step_index == 2
        assert outcome.steps[0].transition == "equivalent"
        assert outcome.steps[1].transition == "partial_overlap"

    def test_later_equivalent_step_does_not_clear_an_earlier_partial_overlap(self) -> None:
        outcome = _run(
            "(x-1)*(x-2)=0",
            ["(x-2)*(x-3)=0", "(x-2)*(x-3)=0"],
        )
        assert outcome.first_invalid_step_index == 1
        assert outcome.status == "invalid_step_found"
        assert outcome.steps[1].transition == "equivalent"


class TestFinalAnswer:
    def test_correct_final_answer_with_invalid_intermediate_step(self) -> None:
        outcome = _run("2*x+3=11", ["2*x=8", "x=5"], ["5"])
        assert outcome.status == "invalid_step_found"
        assert outcome.final_answer is not None
        assert outcome.final_answer.status == "verified_incorrect"

    def test_incorrect_final_answer_after_equivalent_steps(self) -> None:
        outcome = _run("2*x+3=11", ["2*x=8"], ["99"])
        assert outcome.status == "all_steps_equivalent"
        assert outcome.final_answer is not None
        assert outcome.final_answer.status == "verified_incorrect"

    def test_incomplete_quadratic_final_answer(self) -> None:
        outcome = _run("x^2-1=0", [], ["1"])
        assert outcome.final_answer is not None
        assert outcome.final_answer.status == "incomplete"

    def test_empty_solution_set_for_no_real_roots(self) -> None:
        outcome = _run("x^2+1=0", [], [])
        assert outcome.final_answer is not None
        assert outcome.final_answer.status == "verified_correct"
        assert outcome.final_answer.complete is True

    def test_irrational_solution_set(self) -> None:
        outcome = _run("x^2-2=0", [], ["sqrt(2)", "-sqrt(2)"])
        assert outcome.final_answer is not None
        assert outcome.final_answer.status == "verified_correct"

    def test_no_final_answer_submitted_leaves_it_none(self) -> None:
        outcome = _run("2*x=8", ["x=4"], None)
        assert outcome.final_answer is None

    def test_final_answer_checked_against_original_not_last_step(self) -> None:
        # The student's last written step is wrong (x=5), but the final
        # answer field must still be judged against the ORIGINAL equation,
        # not against "x=5" — 4 is correct for 2*x+3=11, 5 is not.
        outcome = _run("2*x+3=11", ["2*x=8", "x=5"], ["4"])
        assert outcome.final_answer is not None
        assert outcome.final_answer.status == "verified_correct"


class TestSecurityAndLimits:
    def test_python_code_injection_in_a_step(self) -> None:
        outcome = _run("2*x=8", ['__import__("os")=1', "x=4"])
        assert outcome.steps[0].transition == "unsupported"
        assert outcome.status == "verification_incomplete"

    def test_arbitrary_function_call_in_a_step(self) -> None:
        outcome = _run("2*x=8", ["x.bit_length()=1"])
        assert outcome.steps[0].transition == "unsupported"

    def test_variable_dependent_denominator_in_a_step(self) -> None:
        outcome = _run("2*x=8", ["1/x=1"])
        assert outcome.steps[0].transition == "unsupported"

    def test_unsupported_degree_in_a_step(self) -> None:
        outcome = _run("2*x=8", ["x^3=8"])
        assert outcome.steps[0].transition == "unsupported"

    def test_unsupported_original_equation_checks_no_steps(self) -> None:
        outcome = _run("x^3-x=0", ["x=0"])
        assert outcome.status == "unsupported"
        assert outcome.steps == []

    def test_malformed_intermediate_equation(self) -> None:
        outcome = _run("2*x=8", ["x"])
        assert outcome.steps[0].transition == "unsupported"

    def test_continuity_lost_then_recovered(self) -> None:
        outcome = _run("2*x=8", ["x + y = 1", "x=4"])
        assert outcome.steps[0].transition == "unsupported"
        assert outcome.steps[1].transition == "unsupported"
        assert "continuity" in outcome.steps[1].detail
        assert outcome.status == "verification_incomplete"
        assert outcome.chain_preserves_solution_set is None

    def test_a_later_clean_comparison_does_not_certify_the_whole_derivation(self) -> None:
        # Step 1 breaks continuity; step 2 recovers (itself unsupported,
        # since it can't be compared to anything); step 3 compares
        # cleanly against step 2's own solution set and is "equivalent".
        # That local cleanliness must not retroactively certify the full
        # chain back to the original equation.
        outcome = _run("2*x=8", ["x + y = 1", "x=4", "x=4"])
        assert outcome.steps[0].transition == "unsupported"
        assert outcome.steps[1].transition == "unsupported"
        assert outcome.steps[2].transition == "equivalent"
        assert outcome.status == "verification_incomplete"
        assert outcome.chain_preserves_solution_set is None
        assert outcome.first_non_equivalent_step_index is None

    def test_an_earlier_invalid_step_is_not_cleared_by_later_steps(self) -> None:
        outcome = _run("2*x+3=11", ["x=5", "x + y = 1", "x=5"])
        assert outcome.steps[0].transition == "incompatible"
        assert outcome.first_invalid_step_index == 1
        assert outcome.status == "invalid_step_found"
        # Later continuity loss and a "locally fine" recovery step must
        # not change the status, the index, or the chain verdict already
        # established by the earlier provable error.
        assert outcome.steps[1].transition == "unsupported"
        assert outcome.steps[2].transition == "unsupported"
        assert outcome.chain_preserves_solution_set is False

    def test_undecidable_symbolic_comparison_is_indeterminate(self, monkeypatch) -> None:
        condition_set = sympy.ConditionSet(X, sympy.Eq(sympy.cos(X), X), sympy.S.Reals)
        monkeypatch.setattr(
            "axiom.verification.engine.solve_real",
            lambda lhs, rhs, var: condition_set if rhs == sympy.Integer(4) else sympy.FiniteSet(4),
        )
        outcome = _run("2*x=8", ["x=4"])
        assert outcome.steps[0].transition == "indeterminate"
        assert outcome.deterministic is False
        assert outcome.status == "verification_incomplete"
