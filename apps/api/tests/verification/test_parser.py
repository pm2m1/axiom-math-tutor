import pytest
import sympy

from axiom.verification.parser import (
    MAX_EXPONENT,
    ParseError,
    parse_calculus_expression,
    parse_candidate,
    parse_equation,
)

X = sympy.Symbol("x")


class TestParseEquationAccepts:
    def test_simple_linear(self) -> None:
        lhs, rhs = parse_equation("2*x + 1 = 7", "x")
        assert lhs == 2 * X + 1
        assert rhs == 7

    def test_quadratic_with_caret_exponent(self) -> None:
        lhs, rhs = parse_equation("x^2 - 5*x + 6 = 0", "x")
        assert lhs == X**2 - 5 * X + 6
        assert rhs == 0

    def test_double_star_exponent(self) -> None:
        lhs, rhs = parse_equation("x**2 - 2 = 0", "x")
        assert lhs == X**2 - 2

    def test_parentheses_and_unary_minus(self) -> None:
        lhs, rhs = parse_equation("-(x + 1) = 2", "x")
        assert lhs == -(X + 1)

    def test_rational_coefficient_via_division(self) -> None:
        lhs, rhs = parse_equation("x/2 + 1 = 3", "x")
        assert lhs == X / 2 + 1

    def test_division_by_constant_expression(self) -> None:
        lhs, _ = parse_equation("x/(1+1) = 1", "x")
        assert lhs == X / 2

    def test_whitespace_is_tolerated(self) -> None:
        lhs, rhs = parse_equation("  2*x + 1   =   7  ", "x")
        assert lhs == 2 * X + 1
        assert rhs == 7


class TestParseEquationRejects:
    def test_missing_equals_sign(self) -> None:
        with pytest.raises(ParseError, match="exactly one"):
            parse_equation("2*x + 1", "x")

    def test_two_equals_signs(self) -> None:
        with pytest.raises(ParseError, match="exactly one"):
            parse_equation("1 = 2 = 3", "x")

    def test_second_variable_name(self) -> None:
        with pytest.raises(ParseError, match="unknown symbol"):
            parse_equation("x + y = 1", "x")

    def test_variable_in_denominator(self) -> None:
        with pytest.raises(ParseError, match="denominator"):
            parse_equation("1/x = 1", "x")

    def test_division_by_zero(self) -> None:
        with pytest.raises(ParseError, match="division by zero"):
            parse_equation("x/0 = 1", "x")

    def test_float_literal_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse_equation("1.5*x = 1", "x")

    def test_string_literal_rejected(self) -> None:
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_equation("x = 'hi'", "x")

    def test_arbitrary_function_call_rejected(self) -> None:
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_equation('x = __import__("os")', "x")

    def test_attribute_access_rejected(self) -> None:
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_equation("x.bit_length() = 1", "x")

    def test_subscript_access_rejected(self) -> None:
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_equation("x[0] = 1", "x")

    def test_list_comprehension_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse_equation("x = [i for i in (1, 2)]", "x")

    def test_non_sqrt_function_call_rejected(self) -> None:
        with pytest.raises(ParseError, match="sqrt"):
            parse_equation("x = abs(2)", "x")

    def test_exponent_above_bound_rejected(self) -> None:
        with pytest.raises(ParseError, match="exponent"):
            parse_equation(f"x^{MAX_EXPONENT + 1} = 1", "x")

    def test_variable_exponent_rejected(self) -> None:
        with pytest.raises(ParseError, match="exponent"):
            parse_equation("2^x = 1", "x")

    def test_negative_exponent_rejected(self) -> None:
        with pytest.raises(ParseError, match="exponent"):
            parse_equation("x^-1 = 1", "x")

    def test_huge_exponent_rejected(self) -> None:
        with pytest.raises(ParseError, match="exponent"):
            parse_equation("x^100000 = 1", "x")

    def test_overly_long_integer_literal_rejected(self) -> None:
        with pytest.raises(ParseError, match="too long"):
            parse_equation("x = " + "1" * 20, "x")

    def test_integer_magnitude_bound(self) -> None:
        with pytest.raises(ParseError, match="too large"):
            parse_equation("x = 9999999999", "x")

    def test_overly_long_expression_rejected(self) -> None:
        with pytest.raises(ParseError, match="too long"):
            parse_equation("x + " + "1 + " * 60 + "1 = 1", "x")

    def test_excessive_ast_depth_rejected(self) -> None:
        chain = "x" + " + 1" * 35
        with pytest.raises(ParseError, match="deeply nested"):
            parse_equation(f"{chain} = 1", "x")

    def test_empty_side_rejected(self) -> None:
        with pytest.raises(ParseError, match="empty"):
            parse_equation("= 1", "x")

    def test_malformed_syntax_rejected(self) -> None:
        with pytest.raises(ParseError, match="malformed"):
            parse_equation("x + = 1", "x")

    def test_unsupported_unicode_character_rejected(self) -> None:
        zero_width_space = chr(0x200B)
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_equation(f"x{zero_width_space} = 1", "x")


class TestParseCandidate:
    def test_integer(self) -> None:
        assert parse_candidate("3") == 3

    def test_negative_integer(self) -> None:
        assert parse_candidate("-3") == -3

    def test_sqrt(self) -> None:
        assert parse_candidate("sqrt(2)") == sympy.sqrt(2)

    def test_negative_sqrt(self) -> None:
        assert parse_candidate("-sqrt(2)") == -sympy.sqrt(2)

    def test_rational(self) -> None:
        assert parse_candidate("1/2") == sympy.Rational(1, 2)

    def test_variable_rejected(self) -> None:
        with pytest.raises(ParseError, match="unknown symbol"):
            parse_candidate("x")

    def test_sqrt_of_negative_rejected(self) -> None:
        with pytest.raises(ParseError, match="negative"):
            parse_candidate("sqrt(-1)")

    def test_sqrt_of_zero_allowed(self) -> None:
        assert parse_candidate("sqrt(0)") == 0

    def test_garbage_text_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse_candidate("banana")

    def test_sqrt_with_two_arguments_rejected(self) -> None:
        # The comma itself is outside the allowed character set, so this
        # is rejected before the sqrt-specific argument-count check runs
        # — still correctly rejected, just by an earlier, cheaper layer.
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_candidate("sqrt(2, 3)")


class TestParseCalculusExpression:
    def test_polynomial(self) -> None:
        assert parse_calculus_expression("x^2 + 2*x + 1", "x") == X**2 + 2 * X + 1

    def test_trig_functions_with_variable_argument(self) -> None:
        assert parse_calculus_expression("sin(x)", "x") == sympy.sin(X)
        assert parse_calculus_expression("cos(x)", "x") == sympy.cos(X)
        assert parse_calculus_expression("tan(x)", "x") == sympy.tan(X)

    def test_exp_and_log_with_variable_argument(self) -> None:
        assert parse_calculus_expression("exp(x)", "x") == sympy.exp(X)
        assert parse_calculus_expression("log(x)", "x") == sympy.log(X)

    def test_sqrt_with_variable_argument_is_allowed_here(self) -> None:
        assert parse_calculus_expression("sqrt(x)", "x") == sympy.sqrt(X)

    def test_composition(self) -> None:
        assert parse_calculus_expression("exp(x) * cos(x)", "x") == sympy.exp(X) * sympy.cos(X)

    def test_sqrt_of_negative_constant_still_rejected(self) -> None:
        with pytest.raises(ParseError, match="negative"):
            parse_calculus_expression("sqrt(-4)", "x")

    def test_variable_denominator_still_rejected(self) -> None:
        with pytest.raises(ParseError, match="denominator"):
            parse_calculus_expression("1/x", "x")

    def test_unknown_function_rejected(self) -> None:
        with pytest.raises(ParseError, match="only"):
            parse_calculus_expression("abs(x)", "x")

    def test_arbitrary_function_call_rejected(self) -> None:
        with pytest.raises(ParseError, match="unsupported characters"):
            parse_calculus_expression('x = __import__("os")', "x")

    def test_huge_exponent_still_rejected(self) -> None:
        with pytest.raises(ParseError, match="exponent"):
            parse_calculus_expression(f"x^{MAX_EXPONENT + 1}", "x")

    def test_equation_mode_unaffected_sin_still_rejected(self) -> None:
        with pytest.raises(ParseError, match="only sqrt"):
            parse_equation("sin(x) = 1", "x")

    def test_candidate_mode_unaffected_sin_still_rejected(self) -> None:
        with pytest.raises(ParseError, match="only sqrt"):
            parse_candidate("sin(1)")
