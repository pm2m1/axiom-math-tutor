# Deterministic mathematical verification (Phase 2A/2B)

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

## Check My Work (Phase 2B)

`POST /api/v1/check-work` verifies an ordered chain of student equation
steps against the original equation, identifies the earliest step that
is *provably* wrong, and — independently — checks a proposed final
answer against the *original* equation using the exact same `verify()`
Phase 2A already validated. No second parser, no second solver: every
step is parsed by `axiom/verification/parser.py` and solved by
`axiom/verification/engine.py::solve_real()`, exactly as Phase 2A built
them.

### Request / response

```json
POST /api/v1/check-work
{
  "equation": "2*x + 3 = 11",
  "variable": "x",
  "steps": ["2*x = 8", "x = 5"],
  "solutions": ["4"]
}
```

```json
{
  "status": "invalid_step_found",
  "steps": [
    {"index": 1, "equation": "2*x = 8", "category": "linear",
     "transition": "equivalent", "detail": "this step preserves the real solution set"},
    {"index": 2, "equation": "x = 5", "category": "linear",
     "transition": "incompatible",
     "detail": "this step changes the solution set incompatibly with the previous equation"}
  ],
  "first_invalid_step_index": 2,
  "first_non_equivalent_step_index": 2,
  "chain_preserves_solution_set": false,
  "deterministic": true,
  "detail": "a provably incorrect step was found",
  "final_answer": {
    "status": "verified_correct", "category": "linear", "domain": "real",
    "deterministic": true, "reason": null,
    "detail": "the proposed solution set matches the exact real solution set",
    "complete": true, "candidates": [{"value": "4", "valid": true}]
  }
}
```

The original equation is step 0 and is not itself in `steps`; student
steps are numbered from 1. `steps` defaults to `[]` (max 15, each
≤200 characters, same restricted grammar as `/api/v1/verify`).
`solutions` distinguishes **omitted** (`null`/absent — no final answer
submitted, `final_answer` is `null`) from an **explicit empty list**
(`[]` — the student's claim of "no real solutions," the same convention
`/api/v1/verify` already uses). Note in the example above: the final
answer (`"4"`) is checked against the *original* equation, not against
the student's last written step (`x = 5`) — it is correct even though
step 2 was wrong, because these are two independent dimensions (see
"Two independent dimensions" below).

### Per-step transition kinds

Each consecutive pair of equations (step *i-1* → step *i*) is compared
by exact SymPy real solution set, via `classify_transition()`:

| `transition` | Meaning |
|---|---|
| `"equivalent"` | Same real solution set exactly — the gold-standard case. |
| `"narrowing"` | New set is a nonempty proper subset of the old one — some solutions kept, others dropped. A legitimate single-branch choice (e.g. `x^2 = 1` → `x = 1`), not an error on its own, but the chain is no longer complete. |
| `"broadening"` | Old set is a proper subset of the new one — every old solution survives, but new ones appeared (e.g. `x = 1` → `x^2 = 1`) that still need independent checking against the original equation. |
| `"partial_overlap"` | Shares some elements with the previous set but is a subset of neither — drops at least one previously valid solution *and* introduces at least one new one. Not a safe branch choice the way pure narrowing is. |
| `"incompatible"` | Disjoint from the previous set (including losing every solution outright) — nothing of the previous solution set survived. The only transition this reports as **provably wrong**. |
| `"unsupported"` | This step's own equation couldn't be parsed or classified under the restricted grammar, or verification continuity was already lost at an earlier step. |
| `"indeterminate"` | This step's own real solution set couldn't be exactly decided by SymPy. |

Only `"incompatible"` sets `first_invalid_step_index`. All four decidable
non-equivalent kinds (`narrowing`/`broadening`/`partial_overlap`/
`incompatible`) set `first_non_equivalent_step_index`, since each is a
*confirmed* departure from exact equivalence; `"unsupported"`/
`"indeterminate"` do not, because those mean "we don't know," not "we
know it's different."

### First-error indexing and continuity

`first_invalid_step_index` is the transition that introduces a *provable*
error — not merely the first step whose written form happens to differ
from the previous one (an equivalent rearrangement, e.g. moving a term
across the `=`, is still `"equivalent"`). A correct final answer never
retroactively validates an earlier incorrect step, and an earlier
non-equivalent (but not incompatible) step never, by itself, marks a
later step wrong — each transition is judged only against its immediate
predecessor's exact solution set.

If a step's own equation can't be parsed/classified, or its solution set
can't be exactly decided, verification continuity is lost at that point:
that step is reported `"unsupported"`/`"indeterminate"` (never silently
skipped and the next step validated as if nothing happened), and the
*next* step — if its own equation does parse and solve cleanly — is
reported `"unsupported"` with an explicit "verification continuity was
lost at an earlier step" detail, but resumes being compared against
*its own* solution set for everything after it. One bad step doesn't
permanently poison the rest of the chain, but the specific gap is never
hidden.

`chain_preserves_solution_set` is `true` only if every step was exactly
`"equivalent"`; `false` as soon as any step is a *confirmed*
non-equivalent kind (regardless of unrelated continuity gaps elsewhere);
`null` when the only problems were continuity gaps with no confirmed
non-equivalence ever observed either way — genuinely unknown, not
assumed true.

### Two independent dimensions

Step-reasoning correctness (`steps`, `first_invalid_step_index`,
`chain_preserves_solution_set`) and final-answer correctness
(`final_answer`) are deliberately kept separate, computed independently,
and never collapsed into one boolean:

- A student can make an invalid algebraic step and still submit the
  correct final answer (by guessing, or by a later unrelated correction)
  — `final_answer.status` can be `"verified_correct"` while
  `status` is `"invalid_step_found"`.
- A sequence of exactly equivalent steps can still end with an
  incomplete or wrong final answer — `status` can be
  `"all_steps_equivalent"` while `final_answer.status` is
  `"incomplete"` or `"verified_incorrect"`.

The API never discloses the complete correct solution set merely
because a step or the final answer was wrong — `final_answer` reports
exactly what `/api/v1/verify` already would for the same equation and
candidates, nothing more. Revealing the answer on request is a
tutoring-policy decision, deferred to a later phase.

### Status meaning

| `status` | Meaning |
|---|---|
| `"all_steps_equivalent"` | Every step preserved the original real solution set exactly. |
| `"valid_but_incomplete"` | No provably incorrect step was found, but at least one step narrowed, broadened, or partially overlapped instead of preserving the set exactly. |
| `"invalid_step_found"` | At least one step was confirmed incompatible with its predecessor. See `first_invalid_step_index`. |
| `"unsupported"` | The *original* equation itself is outside the supported grammar — no steps could be checked at all (`steps` is `[]`). |
| `"verification_incomplete"` | At least one individual step broke continuity (unparseable/unclassifiable/undecidable); later steps may still have been checked against each other once a decidable step reappeared. |

### Frontend

`apps/web/src/components/CheckWorkForm.tsx` — equation input, an
add/remove list of step inputs, an optional add/remove list of
final-answer inputs (with an explicit "this equation has no real
solutions" action distinct from never adding any), loading/duplicate-
submission prevention (the submit button disables while a request is in
flight, mirroring Tutor mode's pattern), and a results view: a numbered
list of per-step transitions with a text label for every kind (never
color alone), the first provably incorrect step flagged with an explicit
"First error" marker, and a separate final-answer panel. Practice mode
remains disabled with its existing note; Tutor mode is untouched.

## Relationship to Phase 1

This remains a separate, additive capability. It does **not**
retroactively make Phase 1's LLM-generated explanations
(`POST /api/v1/answer`) "verified" — nothing calls this engine from the
Phase 1K LangGraph workflow, and no existing route, schema, or behavior
changed. Tutor mode and the feedback system are untouched.

## Known limitations

- **Input-size bounds are not a CPU/memory isolation guarantee.** The
  parser rejects oversized/overly-deep input before any SymPy call, and
  the supported grammar (degree ≤ 2, bounded coefficients, ≤15 steps)
  keeps `solveset`/`Poly` calls cheap in practice — but there is no
  process isolation or hard wall-clock timeout in this milestone. A
  pathological input that passes the stated bounds and still runs
  CPU-bound SymPy work is a theoretical possibility this phase does not
  fully rule out; measured latency is ~3 ms/call for a single
  `/api/v1/verify` equation and ~8–20 ms/call for a representative
  `/api/v1/check-work` chain, far from a concern at this scope, but this
  is not the same claim as "cannot be made to run long," which would
  require process-level isolation this milestone does not add.
- **`sqrt` is only available on proposed-solution values and inside
  equations as a literal of a constant** (e.g. `sqrt(2)`) — it is not a
  general irrational-coefficient feature, and an equation with `sqrt(x)`
  in it will not classify as polynomial (correctly `"unsupported"`).
- **A step that algebraically reduces to a pure numeric statement with
  no trace of the variable** (e.g. simplifying `2*x = 2*x` down to
  `0 = 0`, or `3*x = 3*x + 1` down to `0 = 1`) **is `"unsupported"` for
  that step**, by the same rule `classify()` already applies to the
  original equation (Phase 2A) — there is nothing to classify a
  variable's solution set for once the variable is gone. This is a
  common, legitimate simplification a student might write; Axiom can
  only say "verification continuity was lost here," not confirm or deny
  it, and resumes from the next step that still mentions the variable.
- **No named-misconception diagnosis, no partial credit, no "how close
  was this" scoring.** A transition is one of seven exact
  classifications; there is no notion of a near-miss or a specific
  identified arithmetic slip beyond what the transition label and its
  short detail already say. Mapping a wrong step to a named
  misconception taxonomy is explicitly deferred to a later phase, and no
  LLM is used anywhere in this error-localization path.
- **Classification identifies *what changed about the solution set*, not
  *which arithmetic operation produced it* or *why*.** Two different
  mistakes that happen to produce the same resulting solution set are
  indistinguishable from each other; Axiom reports the set-level
  consequence, never a claimed proof of the specific operation the
  student performed.
- **No mypy stubs for SymPy** — `pyproject.toml` ignores missing SymPy
  type information the same way it already does for `rank_bm25` and
  `scikit-learn`; this is a project-wide pattern, not new risk specific
  to this module.
