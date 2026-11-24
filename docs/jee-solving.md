# JEE-style solving, adaptive tutoring, and benchmark methodology (Phase 3)

This phase combines what the roadmap had sketched as three future
phases — expanded mathematical reasoning, adaptive tutoring, and a real
benchmark — into one milestone, following the brief's own instruction
not to split it further. It does **not** make Axiom a general JEE
Main/Advanced solver. What it adds is a bounded set of new deterministic
solvers, a narrow (and narrowly-scoped) way to reach them from Tutor
mode, adaptive-tutoring features built on Phase 2B's existing
deterministic findings, and a benchmark *harness* — populated with
synthetic, clearly-labeled test data, not real exam content, for a
documented reason below.

## Supported mathematical categories

All four reuse the restricted parser (`axiom/verification/parser.py`,
extended — see "Parser extension" below) and the same exact-computation
philosophy as Phase 2A/2B: compute via SymPy, verify the student's own
claim independently, never trust a textual/structural match.

| Category | Endpoint | Computed via | Verified via |
|---|---|---|---|
| Linear/quadratic equations | `/api/v1/verify`, `/api/v1/check-work` | `solveset` | exact candidate-vs-solution-set comparison (Phase 2A/2B, unchanged) |
| Derivatives | `/api/v1/verify-derivative` | `sympy.diff` | exact symbolic equality of the proposed derivative against `diff()` |
| Indefinite integrals | `/api/v1/verify-integral` | `sympy.integrate` | differentiating the *proposed* antiderivative back and checking it exactly reproduces the original expression — not a textual match against SymPy's own `integrate()` result, since multiple closed forms can be equally correct (e.g. differing by a constant from a trig identity) |
| Combinatorics/probability | `/api/v1/count` | `sympy.binomial`/`factorial`, exact `Rational` | exact rational equality against a proposed answer |

**Deliberately not implemented this phase**: definite integrals (bounds
and singularities need real domain analysis this milestone doesn't
add), general trigonometric *equation* solving with periodicity (`sin(x)
= 1/2` → infinitely many solutions on a periodic pattern — deciding and
presenting a general solution exactly is a materially harder problem
than the bounded equation-solving this engine already does; derivatives
and integrals *of* trig expressions are supported, equation-solving
*with* trig functions is not), and coordinate geometry (lines/circles).
These are honest scope boundaries, not silent gaps — each was weighed
against "a smaller number of correctly implemented, independently
testable categories" and left out rather than done superficially.

### Parser extension

`axiom/verification/parser.py::parse_calculus_expression()` is a new,
explicitly separate entry point: it additionally whitelists `sin`/
`cos`/`tan`/`exp`/`log` alongside `sqrt`, and — unlike everywhere else
in the module — their arguments (and `sqrt`'s) *may* depend on the
variable (`sin(x)`, `sqrt(x)`), since that's the entire point of a
calculus expression. `parse_equation()`/`parse_candidate()` (Phase
2A/2B) are **completely unchanged**: same grammar, same bounds, same
rejections. Division by a variable-dependent denominator remains
unsupported in both modes — calculus support is bounded to polynomial/
trig/exp/log *compositions*, not general rational functions. Still no
`eval`/`exec`/`sympy.sympify`-on-a-string/`parse_expr` anywhere.

### Computed vs. verified vs. unverified

Kept strictly distinct everywhere in this phase:

- **Computed**: SymPy produced a value (`diff`, `integrate`,
  `binomial`/`factorial`) — this is a fact about what the engine
  calculated, not yet a claim about a student's answer.
- **Verified**: an independent deterministic check established a
  specific claim exactly — e.g. "the proposed derivative equals
  `diff(f, x)`," checked by exact symbolic subtraction-and-decide, not
  sampled numerically anywhere in this codebase.
- **Unverified**: Tutor mode's existing LLM-generated explanations
  (`POST /api/v1/answer`, `status: "kb"`/`"web"`) remain exactly what
  Phase 1K already documented them as — grounded, cited, but not
  formally verified. Nothing in this phase changes that or blurs the
  line; `AnswerResponse.verification_method` is now an explicit field
  precisely so a client never has to guess which kind of answer it got.

## Integration into Tutor mode: the deterministic problem router

`axiom/problem_router.py` adds one new node to the existing Phase 1K
LangGraph workflow (`classify_math_problem`, inserted between
`validate_input` and `retrieve_evidence`): a small, explicit, regex-based
matcher for a handful of unambiguous **imperative commands** —
`"differentiate <expr>"`, `"d/dx <expr>"`, `"integrate <expr>"`,
`"<n> choose <r>"`, `"C(n,r)"`, `"P(n,r)"`. A match short-circuits
straight to `finalize_response` with an exactly-computed, exactly-
verified answer (`status: "computed"`); anything that doesn't match —
including the deliberately ambiguous natural question *"What is the
derivative of sin(x)?"*, used elsewhere in this project's own test
suite as a generic example question — takes the identical
`retrieve_evidence` path the graph already had. No existing routing
decision changes for a question the router doesn't recognize; this was
verified by re-running the full pre-existing `test_workflow.py`/
`test_answer_routing.py` suites unchanged and confirming all pass.

**Why not NLP-extract structured math from any free-text question?**
Because the only two ways to do that are (a) trust an LLM to extract
"the expression the student means," which could silently misread the
question — and *that* extraction would itself be unverified, quietly
reintroducing exactly the LLM-trust problem this whole verification
effort exists to avoid — or (b) build real natural-language
understanding, which is a materially larger effort than this milestone.
The router's narrowness is a deliberate, stated boundary: recall is
sacrificed for precision. A missed computation request still gets a
reasonable (if unverified) Tutor-mode answer; a wrongly-intercepted
conceptual question would instead get a terse computed number with no
explanation, which is worse.

One existing file needed a small, additive fix for this to work:
`axiom/guardrails.py`'s math-vocabulary whitelist didn't include the
imperative verbs "differentiate"/"integrate" (only the noun forms
"derivative"/"integral" were present) — without the fix, the guardrail
would reject the router's own command syntax before the router ever
saw it. Added "differentiate", "integrate", "combinatorics" to the
existing whitelist; no existing guardrail test's expectations changed.

## Adaptive tutoring: misconceptions and hints

Both built entirely on Phase 2B's existing deterministic step-transition
classification (`axiom/verification/engine.py::classify_transition`) —
no LLM, no new transition kinds.

**Misconception labels** (`classify_misconception()`): a transition is
labeled only when the label is either true by definition or backed by
one additional exact check:

| Transition | Label | Why this isn't a guess |
|---|---|---|
| `narrowing` | `lost_solution_branch` | True by definition — narrowing *is* dropping solutions |
| `broadening` | `introduced_extraneous_root` | True by definition — broadening *is* gaining unverified ones |
| `incompatible`/`partial_overlap`, and every element of the old solution set negated equals the new set | `sign_error` | A checked pattern (`FiniteSet(*(-e for e in old))  == new`), not inferred from the mismatch alone |
| Anything else non-equivalent | `general_error` | Confirmed wrong, but nothing licenses a more specific claim |

The brief is explicit that a mismatch alone must not be read as a
diagnosis of intent — `general_error` is the honest fallback, not a
rare exception. A full named-misconception taxonomy with broader
coverage is future tutoring-policy work, not approximated here.

**Hint ladder** (`generate_hint_ladder()`, 4 levels): level 1
(conceptual, fixed per-category text), level 2 (directional, fixed
per-category text), level 3 (a verified *fact* about the solution
structure — e.g. "this equation has exactly 2 real solution(s)" — never
the values themselves), level 4 (the actual solution values, from the
already-computed `true_set`). **Level 4 is never included unless the
caller explicitly requests it** — `CheckWorkRequest.hint_level` (1–4)
slices `generate_hint_ladder()`'s output to exactly that many levels;
there is no way to receive level 4 without asking for level 4
specifically. Hints describe the *original* equation (step 0), not
whichever step the student is currently stuck on — a deliberate
simplification, not a claim that every hint is maximally targeted.

## Student mastery model

`axiom/mastery.py` — SQLite via stdlib, the exact pattern
`axiom/feedback.py` already established (no new dependency). Full
design rationale is in that module's docstring; summarized:

- `learner_id` is a client-generated opaque string (a random id in
  browser `localStorage` — see `PracticeForm.tsx`). No authentication,
  no account, no guarantee of identity across devices/sessions — a
  pseudonymous local bucket, explicitly not an account system.
- Mastery is updated **only** from a decisive deterministic outcome
  (`verified_correct`/`verified_incorrect` from Check My Work's
  final-answer check or Practice's grading) — never from an LLM answer,
  a thumbs-up rating, a hint request, or any `"incomplete"`/
  `"unsupported"`/`"indeterminate"` result.
- Update rule: `new = old + 0.3 * (outcome − old)`, outcome `1.0`/`0.0`,
  starting prior `0.5`, clamped to `[0, 1]` — a simple, explainable EMA,
  not a claimed learning-science result.
- Every attempt (including non-updating ones) is logged to an
  `attempts` table for audit/history, separate from the `mastery`
  table's current estimate.
- Survives a backend restart (plain SQLite file, same as feedback.py).

## Practice mode

`axiom/practice.py` — a **small, hand-designed template bank**, not an
LLM generator:

| Skill | Difficulty | Template | Verified property |
|---|---|---|---|
| `linear_equations` | `easy` | `a*x + b = c`, small integers | exactly one integer root |
| `quadratic_equations` | `medium` | `(x-p)*(x-q) = 0` expanded, small integer `p`,`q` | two rational roots |
| `quadratic_equations` | `hard` | `x^2 - k = 0`, `k` a small non-perfect-square | two irrational roots |

Every generated problem is independently re-parsed and re-solved
through `axiom/verification/engine.py` (`classify`/`solve_real`/
`is_decidable_set`) before ever being returned — up to
`MAX_GENERATION_ATTEMPTS` (10) retries; `None` (surfaced as a 503, never
a silently-wrong problem) if that somehow never succeeds. Submitted
answers are graded by the exact same `verify()` function
`/api/v1/verify` uses. `select_next_problem()` is a simple, documented,
explicitly non-adaptive-in-any-statistical-sense heuristic: practice
the weaker skill; escalate quadratic difficulty once mastery clears
0.6. Hints reuse `/api/v1/check-work`'s existing hint mechanism — no
new backend hint endpoint was added.

## Benchmark provenance

**What was attempted.** A real search was made for a legitimate way to
build a genuine JEE Main/Advanced benchmark: NTA (the exam's
administering body) does publish official question papers and answer
keys on its own domains (`nta.ac.in`, `jeemain.nta.nic.in`) after each
session. NTA's own published copyright policy, however, states that
material may be reproduced "free of charge after taking proper
permission from NTA," subject to accurate, non-derogatory use with the
source acknowledged — and explicitly that this does **not** extend to
material identified as copyrighted without separate authorization.
There was no way to obtain that permission within this session, and no
way to independently verify that NTA's public release of a paper for
candidate review also grants a blanket right to redistribute it inside
a third-party repository.

**What was decided, and why.** Rather than (a) committing real,
copyrighted exam content without clear redistribution rights, or
(b) inventing JEE-sounding questions and mislabeling them as authentic
— both explicitly forbidden by this phase's instructions — the harness
*infrastructure* was built and populated with a small, hand-written,
JEE-*style* dataset that is unambiguously labeled as synthetic at every
level: the filenames (`jee_style_synthetic_{dev,frozen}.jsonl`), every
record's `"source": "synthetic"` field, and this document. **No number
in this document or anywhere else in this repository should be read as
measuring performance on real JEE Main/Advanced questions.** If a
properly-licensed dataset becomes available later, it can be loaded
through the exact same `axiom/benchmark/harness.py` loader without any
code change — only the data file changes.

**What the harness actually measures.** Given there is no end-to-end
pipeline that reads a real JEE word problem and extracts this phase's
structured input automatically (see "Integration into Tutor mode"
above for why), a "benchmark" of *that* doesn't exist to run. What
`axiom/benchmark/harness.py` does measure is whether the deterministic
solvers/verifiers themselves are reliable: each item is submitted once
with its hand-verified correct answer (expecting `"verified_correct"`)
and once with a hand-verified incorrect one (expecting
`"verified_incorrect"`), plus a few deliberately out-of-scope items
(expecting `"unsupported"`). For an **exact** symbolic engine, accuracy
should be 100% and the false-verification rate exactly 0% — this is a
reliability/regression check on the verifier, not a performance score
with inherent uncertainty the way an LLM benchmark would have.

**Development discipline.** `jee_style_synthetic_dev.jsonl` (10 items,
19 checks) was used throughout development to catch mistakes.
`jee_style_synthetic_frozen.jsonl` (12 items, 23 checks) was run
exactly once per described change: the first run surfaced two real
*authoring* bugs in the frozen file itself (an identity equation
labeled with a finite "correct" answer that the engine correctly
refused to call complete, and an "incorrect" answer that was actually
one of two valid roots) — both fixed in the dataset before any number
from it was used anywhere, which is the point of hand-checking a
frozen set before relying on it, not "tuning against the holdout."

**Reproduction:**

```bash
cd apps/api
uv run python -m axiom.benchmark.harness --dataset dev
uv run python -m axiom.benchmark.harness --dataset frozen
```

**Actual measured results** (this session, local, zero Meta/Tavily
calls — pure SymPy):

| Dataset | Checks | Accuracy | False-verification rate | Unsupported correctly identified | Median latency | p95 latency |
|---|---|---|---|---|---|---|
| dev | 19 | 100% | 0% | 100% | ~2 ms | ~57 ms |
| frozen | 23 | 100% | 0% | 100% | ~4 ms | ~54 ms |

No comparison of "solver-only vs. LLM-only vs. integrated" accuracy is
reported, because there is no integrated end-to-end JEE-solving
pipeline to compare against (the router only catches explicit
commands, not word problems) — reporting such a comparison would imply
a system that doesn't exist.

## Known limitations (this phase, specifically)

- No definite integrals, no general trigonometric equation solving, no
  coordinate geometry — explicit scope boundaries, see above.
- The problem router only recognizes explicit imperative commands, not
  JEE-style word problems in natural language — full end-to-end
  JEE-question solving is not implemented.
- No genuine, rights-cleared JEE Main/Advanced dataset — see
  "Benchmark provenance." Numbers reported here are a solver-reliability
  check on synthetic data, not a JEE performance claim.
- `select_next_problem()` and the mastery update rule are simple,
  explainable heuristics, not statistically validated adaptive-learning
  algorithms — no claim of pedagogical effectiveness is made or implied.
- No real longitudinal student study backs any "this helps students
  learn" claim — none is made.
- Same CPU/memory-exhaustion caveat as Phase 2A/2B: input bounds (length,
  node count, depth, magnitude) are enforced before any SymPy call, but
  there is no process-level isolation or hard wall-clock timeout in this
  milestone.
