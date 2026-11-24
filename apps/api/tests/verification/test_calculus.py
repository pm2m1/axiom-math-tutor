from axiom.verification.engine import check_derivative, check_integral


class TestCheckDerivative:
    def test_polynomial(self) -> None:
        outcome = check_derivative("x^2", "x", "2*x")
        assert outcome.status == "verified_correct"
        assert outcome.deterministic is True
        assert outcome.computed == "2*x"

    def test_incorrect_polynomial(self) -> None:
        outcome = check_derivative("x^2", "x", "2*x + 1")
        assert outcome.status == "verified_incorrect"

    def test_trig(self) -> None:
        assert check_derivative("sin(x)", "x", "cos(x)").status == "verified_correct"
        assert check_derivative("cos(x)", "x", "-sin(x)").status == "verified_correct"

    def test_product_rule_case(self) -> None:
        outcome = check_derivative("exp(x)*cos(x)", "x", "exp(x)*cos(x) - exp(x)*sin(x)")
        assert outcome.status == "verified_correct"

    def test_wrong_sign_rejected(self) -> None:
        assert check_derivative("cos(x)", "x", "sin(x)").status == "verified_incorrect"

    def test_unparseable_expression_is_unsupported(self) -> None:
        outcome = check_derivative("1/x", "x", "1")
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_expression"

    def test_unparseable_proposed_derivative_is_unsupported(self) -> None:
        outcome = check_derivative("x^2", "x", "banana")
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_proposed_derivative"

    def test_injection_attempt_rejected(self) -> None:
        outcome = check_derivative('__import__("os")', "x", "1")
        assert outcome.status == "unsupported"


class TestCheckIntegral:
    def test_polynomial(self) -> None:
        outcome = check_integral("2*x", "x", "x^2")
        assert outcome.status == "verified_correct"

    def test_additive_constant_is_accepted(self) -> None:
        # Differentiating removes any constant — a correct antiderivative
        # plus an arbitrary number must still verify as correct.
        outcome = check_integral("2*x", "x", "x^2 + 5")
        assert outcome.status == "verified_correct"

    def test_incorrect_antiderivative(self) -> None:
        outcome = check_integral("x^2", "x", "x^2/3")
        assert outcome.status == "verified_incorrect"

    def test_trig(self) -> None:
        assert check_integral("cos(x)", "x", "sin(x)").status == "verified_correct"
        assert check_integral("sin(x)", "x", "-cos(x)").status == "verified_correct"

    def test_two_algebraically_different_correct_forms_both_verify(self) -> None:
        # sin(x)*cos(x) integrates to either -cos(2x)/4 or sin(x)^2/2 —
        # these differ by a constant (a trig identity), and both must
        # verify correct since verification is by differentiating back,
        # never by comparing to one canonical computed form.
        a = check_integral("sin(x)*cos(x)", "x", "sin(x)^2/2")
        b = check_integral("sin(x)*cos(x)", "x", "-cos(2*x)/4")
        assert a.status == "verified_correct"
        assert b.status == "verified_correct"

    def test_unparseable_expression_is_unsupported(self) -> None:
        outcome = check_integral("1/x", "x", "1")
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_expression"

    def test_unparseable_proposed_antiderivative_is_unsupported(self) -> None:
        outcome = check_integral("x^2", "x", "banana")
        assert outcome.status == "unsupported"
        assert outcome.reason == "unparseable_proposed_antiderivative"

    def test_computed_field_is_populated_for_reference(self) -> None:
        outcome = check_integral("2*x", "x", "x^2")
        assert outcome.computed is not None

    def test_injection_attempt_rejected(self) -> None:
        outcome = check_integral('__import__("os")', "x", "1")
        assert outcome.status == "unsupported"
