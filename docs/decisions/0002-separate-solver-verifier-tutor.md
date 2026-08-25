# 0002 — Separate solver, verifier, and tutor policy as distinct components

## Status

Accepted

## Context

Most LLM-based "math tutor" applications use a single model call (or a
single chained prompt) to both work out an answer and decide how to present
it to the student. This is simple to build but makes it impossible to
independently test or reason about mathematical correctness versus
pedagogical behavior — a change intended to make hints gentler can silently
change whether the underlying math is right, because both live in the same
prompt.

## Decision

Axiom's architecture treats mathematical reasoning (the solver), correctness
checking (the verifier), and what the student is shown (the tutor policy) as
three distinct components with an explicit interface between them, per
`docs/architecture.md`. The verifier uses deterministic tools (SymPy first)
wherever one exists for the problem class; an LLM is not treated as a
verifier for anything a deterministic tool can check. The tutor policy
receives the verified result as an input but independently decides how much
of it to reveal, based on mode, hint level, and student mastery.

## Rationale

- Verification needs to be trustworthy independent of how well the model was
  prompted that day. A deterministic check is falsifiable in a way "ask the
  model if it's right" is not.
- Tutoring behavior (how much to reveal, when to hint vs. explain) is a
  product/pedagogy decision that should be tunable and testable without
  touching the mathematical reasoning path, and vice versa.
- This split is also what makes the eventual student model and misconception
  diagnosis meaningful: they operate on a verified fact plus a known-invalid
  step, not on an LLM's unverified narrative of what went wrong.

## Consequences

- More components to build and wire together than a single-prompt approach;
  accepted as the cost of the above.
- Problem categories without a deterministic verifier (open-ended proof
  questions, for instance) will need an explicit "unverified" confidence
  state rather than being silently treated as checked. This is called out
  as required behavior in Phase 2 of the roadmap (abstention path), not
  deferred indefinitely.
