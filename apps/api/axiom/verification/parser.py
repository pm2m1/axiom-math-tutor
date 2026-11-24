"""Restricted expression parser for the Phase 2A verification engine.

This is the one place user-controlled strings are turned into SymPy
expressions, and it is deliberately small and default-deny. It never
calls `eval`/`exec`, `sympy.sympify` on a string, or
`sympy.parsing.sympy_parser.parse_expr` — all of those execute or
evaluate arbitrary-enough grammar to be an injection surface. Instead it
uses `ast.parse(..., mode="eval")` only to get a syntax tree (which does
not execute anything) and walks that tree itself, translating each node
to a SymPy expression one at a time. Any node shape it doesn't
explicitly recognize is rejected, not passed through.

Grammar permitted: integer literals (bounded magnitude), the configured
variable name, parentheses, `+ - * /`, unary `+ -`, `**`/`^` with a
bounded nonnegative integer literal exponent, and `sqrt(...)` of a
bounded nonnegative constant. Division is only allowed when the
denominator does not depend on the variable. Nothing else — no
attribute/subscript access, no other calls, no comprehensions, no
string/byte/complex literals, no other names.

`parse_calculus_expression()` (Phase 3) is a second, explicitly separate
entry point for derivative/integral problems: it additionally whitelists
`sin`/`cos`/`tan`/`exp`/`log`, and — unlike everywhere else in this
module — their arguments (and `sqrt`'s) *may* depend on the variable
(`sin(x)`, `sqrt(x)`, ...), since that is the entire point of a calculus
expression. `parse_equation()`/`parse_candidate()` are completely
unchanged by this addition: their grammar, bounds, and rejections are
identical to Phase 2A/2B. Division by a variable-dependent denominator
remains unsupported in both modes — calculus support here is bounded to
polynomial/trig/exp/log *compositions*, not general rational functions.

This bounds what can be *constructed*, not how expensive it is to
reason about afterward — see the module docstring in `engine.py` and
`docs/verification.md`'s "Known limitations" for what this does and
does not guarantee about CPU/memory exhaustion.
"""

import ast
import re
from collections.abc import Callable

import sympy

MAX_EXPR_CHARS = 200
MAX_INT_DIGITS = 12
MAX_INT_MAGNITUDE = 10**9
MAX_EXPONENT = 6
MAX_AST_NODES = 200
MAX_AST_DEPTH = 30

_ALLOWED_CHARS_RE = re.compile(r"^[0-9A-Za-z\s+\-*/^()=]*$")
_LONG_DIGIT_RUN_RE = re.compile(r"\d{" + str(MAX_INT_DIGITS + 1) + r",}")

_EQUATION_FUNCTIONS: dict[str, Callable[[sympy.Expr], sympy.Expr]] = {"sqrt": sympy.sqrt}
_CALCULUS_FUNCTIONS: dict[str, Callable[[sympy.Expr], sympy.Expr]] = {
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "exp": sympy.exp,
    "log": sympy.log,
}


class ParseError(ValueError):
    """Raised for any input outside the permitted grammar. The message
    is always a short, generic, hand-written string — safe to return to
    a caller as-is, never a raw exception repr or stack trace."""


def parse_equation(text: str, variable: str) -> tuple[sympy.Expr, sympy.Expr]:
    """Splits on exactly one `=` and parses each side with `variable` as
    the only permitted name. Returns `(lhs, rhs)`, not an `Eq` — callers
    decide how to combine them."""
    _validate_raw(text)
    parts = text.split("=")
    if len(parts) != 2:
        raise ParseError("expected exactly one '=' sign")
    lhs_text, rhs_text = parts
    allowed = frozenset({variable})
    lhs = _parse_side(lhs_text, allowed, calculus=False)
    rhs = _parse_side(rhs_text, allowed, calculus=False)
    return lhs, rhs


def parse_candidate(text: str) -> sympy.Expr:
    """Parses a single proposed-solution value. No variable name is
    permitted here — a candidate is a value, not an expression in `x`."""
    return _parse_side(text, frozenset(), calculus=False)


def parse_calculus_expression(text: str, variable: str) -> sympy.Expr:
    """Parses a general single-variable expression for derivative/
    integral problems (Phase 3). Unlike `parse_equation`, `sin`/`cos`/
    `tan`/`exp`/`log` are also permitted, and their arguments — and
    `sqrt`'s — may depend on the variable (`sin(x)`, `sqrt(x)`, ...),
    since that is the point of a calculus expression. Everything else
    (bounds, division restrictions, rejected syntax) is unchanged."""
    return _parse_side(text, frozenset({variable}), calculus=True)


def _parse_side(text: str, allowed_names: frozenset[str], *, calculus: bool) -> sympy.Expr:
    _validate_raw(text)
    normalized = text.strip().replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ParseError("malformed expression") from exc

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        raise ParseError("expression is too complex")
    if _depth(tree) > MAX_AST_DEPTH:
        raise ParseError("expression is too deeply nested")

    return _translate(tree, allowed_names, calculus)


def _validate_raw(text: str) -> None:
    if not text or not text.strip():
        raise ParseError("expression is empty")
    if len(text) > MAX_EXPR_CHARS:
        raise ParseError("expression is too long")
    if not _ALLOWED_CHARS_RE.match(text):
        raise ParseError("expression contains unsupported characters")
    if _LONG_DIGIT_RUN_RE.search(text):
        raise ParseError("numeric literal is too long")


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_depth(child) for child in children)


def _translate(node: ast.AST, allowed_names: frozenset[str], calculus: bool) -> sympy.Expr:
    if isinstance(node, ast.Expression):
        return _translate(node.body, allowed_names, calculus)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ParseError("only integer literals are allowed")
        if abs(value) > MAX_INT_MAGNITUDE:
            raise ParseError("integer literal is too large")
        return sympy.Integer(value)

    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise ParseError(f"unknown symbol '{node.id}'")
        return sympy.Symbol(node.id)

    if isinstance(node, ast.UnaryOp):
        operand = _translate(node.operand, allowed_names, calculus)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ParseError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        return _translate_binop(node, allowed_names, calculus)

    if isinstance(node, ast.Call):
        return _translate_call(node, allowed_names, calculus)

    raise ParseError(f"unsupported syntax: {type(node).__name__}")


def _translate_binop(node: ast.BinOp, allowed_names: frozenset[str], calculus: bool) -> sympy.Expr:
    left = _translate(node.left, allowed_names, calculus)

    if isinstance(node.op, ast.Add):
        return left + _translate(node.right, allowed_names, calculus)
    if isinstance(node.op, ast.Sub):
        return left - _translate(node.right, allowed_names, calculus)
    if isinstance(node.op, ast.Mult):
        return left * _translate(node.right, allowed_names, calculus)

    if isinstance(node.op, ast.Div):
        denominator = _translate(node.right, allowed_names, calculus)
        if denominator.free_symbols:
            raise ParseError("the denominator must not depend on the variable")
        if denominator == 0:
            raise ParseError("division by zero")
        return left / denominator

    if isinstance(node.op, ast.Pow):
        exponent_node = node.right
        if not (
            isinstance(exponent_node, ast.Constant)
            and isinstance(exponent_node.value, int)
            and not isinstance(exponent_node.value, bool)
        ):
            raise ParseError("exponents must be nonnegative integer literals")
        exponent = exponent_node.value
        if not (0 <= exponent <= MAX_EXPONENT):
            raise ParseError(f"exponent must be between 0 and {MAX_EXPONENT}")
        return left**exponent

    raise ParseError("unsupported operator")


def _translate_call(node: ast.Call, allowed_names: frozenset[str], calculus: bool) -> sympy.Expr:
    functions = _CALCULUS_FUNCTIONS if calculus else _EQUATION_FUNCTIONS
    if not (isinstance(node.func, ast.Name) and node.func.id in functions):
        allowed_list = ", ".join(sorted(functions))
        raise ParseError(f"only {allowed_list}(...) calls are allowed")
    func_name = node.func.id
    if node.keywords or len(node.args) != 1:
        raise ParseError(f"{func_name}() takes exactly one argument")

    argument = _translate(node.args[0], allowed_names, calculus)

    if func_name == "sqrt":
        if not calculus and argument.free_symbols:
            raise ParseError("sqrt() argument must not depend on the variable")
        # When the argument is a concrete number (always true outside
        # calculus mode; sometimes true inside it, e.g. `sqrt(4)` used
        # within a larger calculus expression) its sign can be checked
        # exactly. When it depends on the variable, the sign generally
        # cannot be decided at parse time — real-domain validity for
        # those arguments is a documented known limitation, not silently
        # assumed to be fine.
        if not argument.free_symbols and argument < 0:
            raise ParseError("sqrt() of a negative number is not supported in the real domain")

    return functions[func_name](argument)
