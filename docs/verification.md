# Deterministic mathematical verification (Phase 2A)

`POST /api/v1/verify` checks a proposed solution set against a
single-variable equation using SymPy exact computation — no LLM, no
network call, additive to every Phase 1 endpoint. It answers a narrower
question than Phase 1's answer generation: not "explain this topic" but
"is this specific candidate solution set exactly correct, exactly
incorrect, or incomplete, for this specific equation." Nothing here
makes Phase 1's generated explanations "verified" — see "Relationship to
Phase 1" below.

## Scope

Single-variable **linear** and **quadratic** equations, integer/rational
coefficients, exact real roots (including simple irrationals expressible
via `sqrt`). The real-number domain only — no complex roots. Everything
else (higher-degree polynomials, multiple variables, inequalities,
calculus, systems of equations) is explicitly `"unsupported"`, never
guessed at.

Classification is by the equation's actual polynomial structure after
exact SymPy simplification (`Poly` degree), not by the highest exponent
that happened to appear in the source text — `x^2 - x^2 + 2*x - 6 = 0`
is linear, not quadratic, because the `x^2` terms cancel exactly.

## Request / response

```json
POST /api/v1/verify
{"equation": "x^2 - 5*x + 6 = 0", "variable": "x", "solutions": ["2", "3"]}
```

```json
{
  "status": "verified_correct",
  "category": "quadratic",
  "domain": "real",
  "deterministic": true,
  "reason": null,
  "detail": "the proposed solution set matches the exact real solution set",
  "complete": true,
  "candidates": [{"value": "2", "valid": true}, {"value": "3", "valid": true}]
}
```

`variable` defaults to `"x"` and must be a single letter. `solutions` is
an empty list by default (max 20 entries) — an empty list is a claim
that the equation has **no real solutions**, not "no answer given."

### Status meaning

| `status` | Meaning |
|---|---|
| `"verified_correct"` | Every proposed value satisfies the equation, and the set is exactly complete. |
| `"verified_incorrect"` | At least one proposed value is wrong, or an empty list falsely claims no solutions exist. |
| `"incomplete"` | Every proposed value is individually valid, but at least one real solution is missing. |
| `"unsupported"` | Out of scope: not single-variable linear/quadratic, not polynomial, or unparseable under the restricted grammar. See `reason`. |
| `"indeterminate"` | In scope, but SymPy could not exactly decide the solution set or a candidate's validity. Never reported as correct or incorrect. |

`deterministic` is `false` only for `"indeterminate"` — every other
status was reached by exact computation (`Poly`, `solveset` over
`S.Reals`, exact symbolic equality via `.equals(0)`), never a
floating-point approximation. `complete` is only meaningful once
candidate validity is established (`None` for `"unsupported"`).
`candidates` reports per-value validity (`valid: null` only under
`"indeterminate"`).

**Checking a candidate is not the same as proving completeness.** A
value can individually satisfy the equation (`valid: true`) while the
overall submission is still `"incomplete"` — `x^2 - 1 = 0` with only
`["1"]` proposed is exactly this case. The API never collapses these
into a single pass/fail bit, and never discloses the full correct
solution set for an incorrect/incomplete submission — correct-solution
disclosure is a tutoring-policy decision deferred to Phase 2B/3, not a
side effect of verification.

## The restricted parser

`axiom/verification/parser.py` is the only place a raw string becomes a
SymPy expression, and it never calls `eval`/`exec`, `sympy.sympify` on a
string, or `sympy.parsing.sympy_parser.parse_expr` — all of those
execute or evaluate arbitrary-enough grammar to be an injection surface.
It uses `ast.parse(..., mode="eval")` only to get a syntax tree (which
does not execute anything) and walks that tree itself, translating one
node at a time; any shape it doesn't explicitly recognize is rejected.

**Grammar**: integer literals, the configured variable, parentheses,
`+ - * /`, unary `+ -`, `^`/`**` with a bounded nonnegative integer
literal exponent, and `sqrt(...)` of a bounded nonnegative constant.
Division is only allowed when the denominator doesn't depend on the
variable.

**Rejected, explicitly**: attribute/subscript access, any function call
other than `sqrt`, comprehensions, string/byte/complex literals, any
name other than the configured variable, a second variable, a
variable-dependent denominator, negative/variable/huge exponents,
`sqrt` of a negative number, overly long numeric literals, and
expressions beyond documented length/node-count/depth bounds.

**Bounds** (all rejected before any SymPy computation runs):

| Limit | Value |
|---|---|
| Expression length | 200 characters |
| Integer literal magnitude | 10^9 |
| Integer literal digit count | 12 |
| Exponent | 0–6 |
| AST node count | 200 |
| AST depth | 30 |

## SymPy verification strategy

Small, independently testable functions (`axiom/verification/engine.py`),
composed by `verify()`:

1. `classify(lhs, rhs, variable)` — `Poly(expand(lhs - rhs), variable).degree()`
   after confirming no other free symbols; degree 1 (or the degenerate
   0/-∞ case, when `variable` appeared in the raw equation and then
   cancelled) → `"linear"`; degree 2 → `"quadratic"`; otherwise `None`
   (unsupported).
2. `solve_real(lhs, rhs, variable)` — `solveset(Eq(lhs - rhs, 0), variable, domain=S.Reals)`.
3. `is_decidable_set(solution_set)` — `True` only for `S.EmptySet`,
   `S.Reals`, or a `FiniteSet` of concrete (symbol-free) numbers; a
   `ConditionSet` or anything else unresolved is `False`.
4. `check_candidate(lhs, rhs, variable, value)` — substitutes into the
   **original** `lhs`/`rhs` (never a moved-to-one-side or otherwise
   transformed form) and calls `.equals(0)`; `None` means SymPy could
   not decide, surfaced as `"indeterminate"`, never assumed.
5. `compare_to_solution_set(true_set, proposed_values)` — exact SymPy
   set difference (`true_set - proposed_set`, `proposed_set - true_set`)
   to decide completeness; duplicates in `proposed_values` are
   deduplicated automatically by `FiniteSet`.

An identity (`2*x = 2*x`, solution set `S.Reals`) and a contradiction
(`3*x = 3*x + 1`, solution set `S.EmptySet`) are both linear and both
exercised explicitly — an empty submitted list is correct against the
second and incorrect against the first, which is exactly the
distinction the project asked this engine to get right.

## Relationship to Phase 1

This is a separate, additive capability. It does **not** retroactively
make Phase 1's LLM-generated explanations (`POST /api/v1/answer`)
"verified" — nothing calls this engine from the Phase 1K LangGraph
workflow, and no existing route, schema, or behavior changed. Tutor mode
and the feedback system are untouched. Check My Work remains disabled in
the frontend with its existing "not available yet" note — wiring this
verifier into it for real step-by-step checking is Phase 2B, not this
milestone.

## Known limitations

- **Input-size bounds are not a CPU/memory isolation guarantee.** The
  parser rejects oversized/overly-deep input before any SymPy call, and
  the supported grammar (degree ≤ 2, bounded coefficients) keeps
  `solveset`/`Poly` calls cheap in practice — but there is no process
  isolation or hard wall-clock timeout in this milestone. A pathological
  input that passes the stated bounds and still runs CPU-bound SymPy
  work is a theoretical possibility this phase does not fully rule out;
  measured latency across the supported test matrix is ~3 ms/call (see
  the Phase 2A report), far from a concern at this scope, but this is
  not the same claim as "cannot be made to run long," which would
  require process-level isolation this milestone does not add.
- **`sqrt` is only available on proposed-solution values and inside
  equations as a literal of a constant** (e.g. `sqrt(2)`) — it is not a
  general irrational-coefficient feature, and an equation with `sqrt(x)`
  in it will not classify as polynomial (correctly `"unsupported"`).
- **No partial credit, no step verification.** A submission is either
  the complete correct set, individually-valid-but-incomplete, or wrong
  — there is no notion of "close" or a diagnosed error location. That is
  Phase 2B/3 (Check My Work, misconception diagnosis).
- **No mypy stubs for SymPy** — `pyproject.toml` ignores missing SymPy
  type information the same way it already does for `rank_bm25` and
  `scikit-learn`; this is a project-wide pattern, not new risk specific
  to this module.
