from axiom.verification.engine import check_combinatorics


class TestNcr:
    def test_computed_value(self) -> None:
        outcome = check_combinatorics("ncr", 5, 2)
        assert outcome.status == "computed"
        assert outcome.computed == "10"

    def test_verified_correct(self) -> None:
        outcome = check_combinatorics("ncr", 5, 2, proposed_answer="10")
        assert outcome.status == "verified_correct"

    def test_verified_incorrect(self) -> None:
        outcome = check_combinatorics("ncr", 5, 2, proposed_answer="9")
        assert outcome.status == "verified_incorrect"

    def test_r_zero(self) -> None:
        assert check_combinatorics("ncr", 5, 0).computed == "1"

    def test_r_equals_n(self) -> None:
        assert check_combinatorics("ncr", 5, 5).computed == "1"

    def test_r_exceeds_n_is_unsupported(self) -> None:
        outcome = check_combinatorics("ncr", 3, 5)
        assert outcome.status == "unsupported"
        assert outcome.reason == "r_exceeds_n"


class TestNpr:
    def test_computed_value(self) -> None:
        outcome = check_combinatorics("npr", 5, 2)
        assert outcome.status == "computed"
        assert outcome.computed == "20"

    def test_verified_correct(self) -> None:
        outcome = check_combinatorics("npr", 5, 2, proposed_answer="20")
        assert outcome.status == "verified_correct"


class TestProbability:
    def test_computed_value(self) -> None:
        outcome = check_combinatorics("probability", 10, 2, favorable=3)
        assert outcome.status == "computed"
        assert outcome.computed == "1/15"

    def test_verified_correct(self) -> None:
        outcome = check_combinatorics("probability", 10, 2, favorable=3, proposed_answer="1/15")
        assert outcome.status == "verified_correct"

    def test_verified_incorrect(self) -> None:
        outcome = check_combinatorics("probability", 10, 2, favorable=3, proposed_answer="1/2")
        assert outcome.status == "verified_incorrect"

    def test_zero_probability_when_r_exceeds_favorable(self) -> None:
        outcome = check_combinatorics("probability", 10, 5, favorable=2)
        assert outcome.status == "computed"
        assert outcome.computed == "0"

    def test_certain_probability_when_all_are_favorable(self) -> None:
        outcome = check_combinatorics("probability", 5, 5, favorable=5)
        assert outcome.computed == "1"

    def test_missing_favorable_is_unsupported(self) -> None:
        outcome = check_combinatorics("probability", 10, 2)
        assert outcome.status == "unsupported"
        assert outcome.reason == "favorable_required"

    def test_favorable_exceeds_n_is_unsupported(self) -> None:
        outcome = check_combinatorics("probability", 5, 2, favorable=10)
        assert outcome.status == "unsupported"
        assert outcome.reason == "favorable_exceeds_n"


class TestCombinatoricsInputHandling:
    def test_unparseable_proposed_answer_is_unsupported(self) -> None:
        outcome = check_combinatorics("ncr", 5, 2, proposed_answer="banana")
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_proposed_answer"

    def test_injection_attempt_in_proposed_answer_rejected(self) -> None:
        outcome = check_combinatorics("ncr", 5, 2, proposed_answer='__import__("os")')
        assert outcome.status == "unsupported"

    def test_irrational_proposed_answer_parses_but_is_incorrect(self) -> None:
        outcome = check_combinatorics("ncr", 2, 1, proposed_answer="sqrt(4)")
        assert outcome.status == "verified_correct"
