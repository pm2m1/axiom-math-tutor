import sympy

from axiom.verification.engine import (
    CandidateResult,
    check_candidate,
    classify,
    compare_to_solution_set,
    is_decidable_set,
    solve_real,
    verify,
)

X = sympy.Symbol("x")
Y = sympy.Symbol("y")


class TestClassify:
    def test_linear(self) -> None:
        assert classify(2 * X + 1, sympy.Integer(7), X) == "linear"

    def test_quadratic(self) -> None:
        assert classify(X**2 - 5 * X + 6, sympy.Integer(0), X) == "quadratic"

    def test_degree_zero_after_cancellation_is_linear(self) -> None:
        assert classify(3 * X, 3 * X + 1, X) == "linear"

    def test_identity_after_cancellation_is_linear(self) -> None:
        assert classify(2 * X, 2 * X, X) == "linear"

    def test_variable_absent_is_unsupported(self) -> None:
        assert classify(sympy.Integer(5), sympy.Integer(3), X) is None

    def test_higher_degree_is_unsupported(self) -> None:
        assert classify(X**3 - X, sympy.Integer(0), X) is None

    def test_other_symbol_is_unsupported(self) -> None:
        assert classify(X + Y, sympy.Integer(1), X) is None


class TestSolveReal:
    def test_linear(self) -> None:
        assert solve_real(2 * X + 1, sympy.Integer(7), X) == sympy.FiniteSet(3)

    def test_quadratic_two_roots(self) -> None:
        assert solve_real(X**2 - 1, sympy.Integer(0), X) == sympy.FiniteSet(-1, 1)

    def test_quadratic_no_real_roots(self) -> None:
        assert solve_real(X**2 + 1, sympy.Integer(0), X) == sympy.S.EmptySet

    def test_quadratic_irrational_roots(self) -> None:
        assert solve_real(X**2 - 2, sympy.Integer(0), X) == sympy.FiniteSet(
            sympy.sqrt(2), -sympy.sqrt(2)
        )

    def test_repeated_root_counted_once(self) -> None:
        result = solve_real(X**2 - 2 * X + 1, sympy.Integer(0), X)
        assert result == sympy.FiniteSet(1)
        assert len(result) == 1

    def test_contradiction_is_empty(self) -> None:
        assert solve_real(3 * X, 3 * X + 1, X) == sympy.S.EmptySet

    def test_identity_is_reals(self) -> None:
        assert solve_real(2 * X, 2 * X, X) == sympy.S.Reals


class TestIsDecidableSet:
    def test_empty_set(self) -> None:
        assert is_decidable_set(sympy.S.EmptySet) is True

    def test_reals(self) -> None:
        assert is_decidable_set(sympy.S.Reals) is True

    def test_finite_set_of_numbers(self) -> None:
        assert is_decidable_set(sympy.FiniteSet(1, sympy.sqrt(2))) is True

    def test_condition_set_is_not_decidable(self) -> None:
        condition_set = sympy.ConditionSet(X, sympy.Eq(sympy.cos(X), X), sympy.S.Reals)
        assert is_decidable_set(condition_set) is False


class TestCheckCandidate:
    def test_valid_candidate(self) -> None:
        assert check_candidate(2 * X + 1, sympy.Integer(7), X, sympy.Integer(3)) is True

    def test_invalid_candidate(self) -> None:
        assert check_candidate(2 * X + 1, sympy.Integer(7), X, sympy.Integer(4)) is False

    def test_irrational_candidate(self) -> None:
        assert check_candidate(X**2, sympy.Integer(2), X, sympy.sqrt(2)) is True

    def test_checks_original_equation_not_a_transformed_one(self) -> None:
        # x/2 = 1 transformed naively (multiply both sides by 2) gives
        # x = 2; a candidate must still be checked against the original.
        assert check_candidate(X / 2, sympy.Integer(1), X, sympy.Integer(2)) is True
        assert check_candidate(X / 2, sympy.Integer(1), X, sympy.Integer(4)) is False


class TestCompareToSolutionSet:
    def test_exact_match_is_complete(self) -> None:
        true_set = sympy.FiniteSet(2, 3)
        result = compare_to_solution_set(true_set, [sympy.Integer(2), sympy.Integer(3)])
        assert result.complete is True
        assert result.mismatch is False
        assert result.missing_count == 0

    def test_partial_match_is_incomplete(self) -> None:
        true_set = sympy.FiniteSet(-1, 1)
        result = compare_to_solution_set(true_set, [sympy.Integer(1)])
        assert result.complete is False
        assert result.missing_count == 1

    def test_duplicates_are_deduplicated(self) -> None:
        true_set = sympy.FiniteSet(2)
        result = compare_to_solution_set(true_set, [sympy.Integer(2), sympy.Integer(2)])
        assert result.complete is True

    def test_mismatch_when_proposed_value_outside_true_set(self) -> None:
        true_set = sympy.FiniteSet(2, 3)
        result = compare_to_solution_set(true_set, [sympy.Integer(2), sympy.Integer(99)])
        assert result.mismatch is True


def _status(equation: str, solutions: list[str]) -> str:
    return verify(equation, "x", solutions).status


class TestVerifyLinear:
    def test_correct(self) -> None:
        outcome = verify("2*x + 1 = 7", "x", ["3"])
        assert outcome.status == "verified_correct"
        assert outcome.category == "linear"
        assert outcome.complete is True
        assert outcome.candidates == [CandidateResult(value="3", valid=True)]

    def test_incorrect(self) -> None:
        outcome = verify("2*x + 1 = 7", "x", ["4"])
        assert outcome.status == "verified_incorrect"
        assert outcome.candidates == [CandidateResult(value="4", valid=False)]

    def test_contradiction_has_no_solutions(self) -> None:
        outcome = verify("3*x = 3*x + 1", "x", [])
        assert outcome.status == "verified_correct"
        assert outcome.category == "linear"

    def test_contradiction_rejects_a_proposed_root(self) -> None:
        outcome = verify("3*x = 3*x + 1", "x", ["5"])
        assert outcome.status == "verified_incorrect"
        assert outcome.reason == "no_real_solutions"

    def test_identity_has_infinitely_many_solutions(self) -> None:
        outcome = verify("2*x = 2*x", "x", [])
        assert outcome.status == "verified_incorrect"
        assert outcome.reason == "infinite_solutions"

    def test_identity_distinguished_from_empty_set(self) -> None:
        empty_claim = verify("2*x = 2*x", "x", [])
        no_solution_claim = verify("3*x = 3*x + 1", "x", [])
        assert empty_claim.status != no_solution_claim.status

    def test_identity_with_a_valid_candidate_is_incomplete(self) -> None:
        outcome = verify("2*x = 2*x", "x", ["3"])
        assert outcome.status == "incomplete"
        assert outcome.reason == "infinite_solutions"


class TestVerifyQuadratic:
    def test_correct_two_roots(self) -> None:
        outcome = verify("x^2 - 5*x + 6 = 0", "x", ["2", "3"])
        assert outcome.status == "verified_correct"
        assert outcome.category == "quadratic"

    def test_incomplete_missing_one_root(self) -> None:
        outcome = verify("x^2 - 1 = 0", "x", ["1"])
        assert outcome.status == "incomplete"
        assert outcome.reason == "missing_solutions"

    def test_correct_both_roots(self) -> None:
        outcome = verify("x^2 - 1 = 0", "x", ["-1", "1"])
        assert outcome.status == "verified_correct"

    def test_no_real_solutions_correct_empty_claim(self) -> None:
        outcome = verify("x^2 + 1 = 0", "x", [])
        assert outcome.status == "verified_correct"
        assert outcome.complete is True

    def test_irrational_roots_correct(self) -> None:
        outcome = verify("x^2 - 2 = 0", "x", ["sqrt(2)", "-sqrt(2)"])
        assert outcome.status == "verified_correct"

    def test_repeated_root_counted_once(self) -> None:
        outcome = verify("x^2 - 2*x + 1 = 0", "x", ["1"])
        assert outcome.status == "verified_correct"
        assert outcome.complete is True


class TestVerifyUnsupportedAndInvalid:
    def test_higher_degree_polynomial(self) -> None:
        outcome = verify("x^3 - x = 0", "x", ["0"])
        assert outcome.status == "unsupported"
        assert outcome.reason == "unsupported_degree"

    def test_multiple_variables(self) -> None:
        outcome = verify("x + y = 1", "x", ["1"])
        assert outcome.status == "unsupported"

    def test_variable_in_denominator(self) -> None:
        outcome = verify("1/x = 1", "x", ["1"])
        assert outcome.status == "unsupported"

    def test_arbitrary_python_execution_attempt(self) -> None:
        outcome = verify('__import__("os") = 1', "x", [])
        assert outcome.status == "unsupported"

    def test_function_or_attribute_access(self) -> None:
        outcome = verify("x.bit_length() = 1", "x", ["1"])
        assert outcome.status == "unsupported"

    def test_huge_exponent(self) -> None:
        outcome = verify("x^100000 = 1", "x", ["1"])
        assert outcome.status == "unsupported"

    def test_excessively_complex_expression(self) -> None:
        outcome = verify("x + " + "1 + " * 60 + "1 = 1", "x", [])
        assert outcome.status == "unsupported"

    def test_invalid_candidate_value(self) -> None:
        outcome = verify("2*x = 4", "x", ["banana"])
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_candidate"

    def test_malformed_equation(self) -> None:
        outcome = verify("2*x", "x", ["3"])
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_equation"

    def test_duplicate_proposed_roots(self) -> None:
        outcome = verify("2*x = 4", "x", ["2", "2"])
        assert outcome.status == "verified_correct"

    def test_non_real_candidate_root_rejected(self) -> None:
        outcome = verify("x^2 + 1 = 0", "x", ["sqrt(-1)"])
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_candidate"

    def test_degenerate_constant_equation(self) -> None:
        outcome = verify("5 = 3", "x", [])
        assert outcome.status == "unsupported"
        assert outcome.reason == "missing_variable"

    def test_missing_solutions_against_a_solvable_equation(self) -> None:
        outcome = verify("2*x + 1 = 7", "x", [])
        assert outcome.status == "verified_incorrect"
        assert outcome.reason == "missing_solutions"


class TestVerifyIndeterminate:
    """The restricted linear/quadratic grammar always resolves exactly in
    practice — these exercise the indeterminate path directly against
    the internal functions, since no input reachable through the public
    parser is expected to trigger it for real."""

    def test_undecidable_solution_set_reported_as_indeterminate(self, monkeypatch) -> None:
        condition_set = sympy.ConditionSet(X, sympy.Eq(sympy.cos(X), X), sympy.S.Reals)
        monkeypatch.setattr("axiom.verification.engine.solve_real", lambda *a, **k: condition_set)
        outcome = verify("x + 1 = 2", "x", [])
        assert outcome.status == "indeterminate"
        assert outcome.deterministic is False

    def test_undecidable_candidate_reported_as_indeterminate(self, monkeypatch) -> None:
        monkeypatch.setattr("axiom.verification.engine.check_candidate", lambda *a, **k: None)
        outcome = verify("x + 1 = 2", "x", ["1"])
        assert outcome.status == "indeterminate"
        assert outcome.candidates == [CandidateResult(value="1", valid=None)]
