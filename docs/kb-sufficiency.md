# KB sufficiency (Phase 1G/1H)

Retrieval always returns nearest neighbours — the chunks most similar to a
query, whether or not they actually explain it. Phase 1A–1F measured
whether the selected retriever (`rank_bm25` + Qdrant dense + application
RRF) finds the *right* chunk when the corpus covers a question. This phase
asks a different, prior question: does the corpus cover the question at
all, and can retrieval evidence alone tell the difference?

The distinction matters concretely. `equations-properties-08` mentions the
Associative Property, the Commutative Property, and the Distributive
Property by name — as a pointer to a different chapter not in this corpus
— without explaining any of them. A query about the distributive property
retrieves that chunk with a real, non-trivial score. That is not
sufficient support. Labeling it supported because a chunk is topically
close would defeat the purpose of this evaluation.

## Methodology correction (Phase 1H)

Phase 1G's `test` split (36 rows) has since been read, and its two false
accepts directly motivated Phase 1H's research question below. That means
it is no longer an untouched holdout and this document stops calling it
one: it is the **historical v1 test set** from here on — preserved exactly
as authored, still reported for comparison, but not a valid estimate of
generalization to unseen queries. Phase 1H's fresh holdout
(`algebra_sufficiency_v2.jsonl`, see below) is the dataset that plays that
role now, and it was evaluated exactly once, only after every modeling
decision in Phase 1H was frozen.

## Why the existing 8 out-of-domain cases were inadequate

Phase 1A's evaluation set has 8 out-of-scope queries (calculus, geometry,
unrelated topics) alongside 42 in-scope ones, and every phase since (1B
through 1F) repeated a confidence investigation against that same 8-query
sample. That's enough to say retrieval scores don't cleanly separate
in-scope from out-of-scope, but 8 examples — all easy, obviously-different
topics — can't test the harder and more important case: an algebra
question the retriever answers *confidently and wrongly*, because the
corpus discusses the topic without actually covering it. That case needs
its own, larger, purpose-built dataset — this phase's benchmark, not a
retrieval-quality benchmark repurposed for it.

## The retrieval benchmark is unchanged

`data/evaluation/algebra_retrieval.jsonl` (50 queries, Phase 1A) was not
modified, relabeled, or reused for this phase. This is a new, separate
dataset: `data/evaluation/algebra_sufficiency.jsonl`.

## Benchmark: `algebra_sufficiency.jsonl`

107 queries against the same 40-chunk corpus, split into three classes:

- **Supported (54)** — genuinely answerable from the corpus: direct
  questions, paraphrases, conceptual questions, worked-problem-style
  questions, and terminology questions, spread across equations,
  inequalities, and quadratics.
- **Near-domain unsupported (36)** — the important group. These look like
  real algebra questions and would retrieve plausible algebra chunks, but
  the corpus does not actually cover them: the distributive/commutative/
  associative/identity properties (named but not explained), the
  zero-product property, factoring, absolute-value *equations* (as opposed
  to the absolute-value *inequalities* the corpus does cover), the
  rational root theorem, the polynomial remainder theorem, polynomial long
  division, systems of linear equations, matrices, complex quadratic
  roots (the discriminant chunk states that they exist but never computes
  one), graphing a parabola's vertex, slope/graphing lines, and general
  exponent rules. One entry, "How do you graph a linear inequality?", is a
  chunk-verified case where a relevant-looking chunk *exists*
  (`inequalities-basics-12`) but is a truncated stub ("A linear inequality
  is written in" cuts directly into navigation text) — a naive retriever
  would confidently match it, but it explains nothing.
- **Easy out-of-domain (17)** — calculus, geometry, probability,
  statistics, physics, finance, and unrelated topics. Deliberately a
  minority of the unsupported set: a router that only detects "this isn't
  algebra" is not what this phase is measuring.

### Labeling protocol

Every supported label's `support_chunk_ids` were checked by reading the
corpus text directly (see `data/corpus/algebra_chunks.jsonl`) — not
generated and assumed correct. Every near-domain negative was checked the
same way in the other direction: the full corpus was read end to end and
confirmed not to explain the referenced concept, beyond passing mentions.
`data/README.md`'s existing documented gaps (distributive property,
factoring, zero-product property, absolute-value equations) were treated
as leads to verify, not facts to trust — each was re-confirmed against the
actual chunk text during this labeling pass, and the `graphing_lines`
stub case above was found this way rather than assumed. Candidate queries
were drafted by topic area, then checked one at a time against this
process; none were generated by an LLM and left unverified.

### Development / test split

Fixed at authoring time, recorded per row in the `split` field — not
re-randomized on each run. 71 development / 36 test, with all three
classes present in both:

| | development | test |
|---|---:|---:|
| supported | 37 | 17 |
| near-domain unsupported | 23 | 13 |
| easy OOD unsupported | 11 | 6 |

Every threshold and the logistic regression model below were selected
using development data only. The test split was evaluated exactly once,
after that selection was fixed.

### Leakage checks

All 107 IDs are unique; all 107 query strings are unique (no duplicate or
near-duplicate text was found across the development/test boundary on
manual inspection); every `support_chunk_ids` entry resolves to a real
corpus chunk ID; every supported case has at least one support ID and
every unsupported case has none; every unsupported case has a valid
`negative_type`. Enforced by `tests/retrieval/test_sufficiency.py` against
the committed file, not just checked once by hand.

## Decision features

From the selected pipeline only — `rank_bm25.BM25Retriever` and
`QdrantDenseRetriever`, both frozen, plus the existing application-side
`HybridRetriever` for one fused score:

- `bm25_top1`, `bm25_margin` (top-1 minus top-2 BM25 score)
- `dense_top1`, `dense_margin` (top-1 minus top-2 cosine similarity)
- `top1_agreement` (do BM25 and dense agree on the top-1 chunk)
- `top5_overlap` (Jaccard overlap of their top-5 chunk sets)
- `rrf_top1` (the fused RRF score, used only for its own threshold
  baseline — Phase 1B–1D already found RRF's score compressed and poorly
  separated; it is not fed into the classifier)

## Baselines

Three scalar thresholds (BM25, dense, RRF top-1 score) plus one logistic
regression over the six classifier features (`bm25_top1`, `bm25_margin`,
`dense_top1`, `dense_margin`, `top1_agreement`, `top5_overlap`) —
standardized, `scikit-learn`, deterministic `random_state=0`, trained on
development only, no query text as input (retrieval evidence only).

### Threshold selection

For every scalar and the classifier's own decision score: sweep every
development score as a candidate threshold; among thresholds reaching KB-
accept precision ≥ 0.90 on development, keep the one with the highest
supported-query recall. 0.90 is a safety-oriented starting target, chosen
before looking at any result, not tuned to flatter one baseline — and the
procedure reports honestly when it isn't reachable rather than forcing it
(this happened for the RRF baseline; see results below).

## Results (historical v1 test, computed once — Phase 1G)

```text
                    threshold   met 0.90?   test precision   test recall   false_accept   false_reject
BM25 threshold      23.35       yes         1.00             0.24          0.00           0.76
Dense threshold      0.78       yes         0.89             0.47          0.05           0.53
RRF threshold        0.03       no          0.62             0.88          0.47           0.12
Logistic regression  0.60       yes         0.83             0.59          0.11           0.41
```

The RRF baseline is a genuine negative control, not a bug: it could not
reach 0.90 precision on development at all (best achievable was 0.66) —
consistent with every earlier phase's finding that RRF's fused score has
poor KB-sufficiency separation. The BM25 threshold reaches perfect test
precision only by being so conservative it rejects almost everything
(recall 0.24, false-reject 0.76) — not a usable operating point despite
the "met 0.90" label. Dense and the logistic regression are the only two
with a real, usable trade-off.

Confusion matrix (logistic regression, historical v1 test, n=36):
`tp=10 fp=2 tn=17 fn=7`. Accuracy 0.75, F1 0.69.

### Hard negatives (historical v1 test)

```text
                     near-domain rejection   easy-OOD rejection
BM25 threshold       1.00                    1.00
Dense threshold      0.92                    1.00
RRF threshold        0.46                    0.67
Logistic regression  0.85                    1.00
```

Every approach rejects easy-OOD perfectly — that's the easy case. The
logistic regression's near-domain rejection (0.85, 11/13) is the number
that actually matters: it shows the classifier is not merely detecting
"this isn't algebra" (RRF is much worse here, at 0.46, despite easy-OOD
parity), but its 2 near-domain false accepts (below) show a real,
systematic weakness, not noise.

## Failure analysis (logistic regression, historical v1 test)

**False accepts** (near-domain unsupported, wrongly used): "What is the
associative property in algebra?" (`bm25_top1=15.2`, `dense_top1=0.78`)
and "What are the complex roots of x^2 + 4 = 0?" (`bm25_top1=10.8`,
`dense_top1=0.71`) both score comparably to genuinely-supported queries.
Both have an identifiable, explainable cause: `equations-properties-08`
names the associative property without explaining it, and
`quadratics-formula-01` states that a negative discriminant gives
"conjugate imaginary solutions" without computing one — both chunks are
lexically and semantically close to their query without being sufficient,
which is exactly the failure mode this benchmark's near-domain group was
built to surface, not a labeling mistake.

**False rejects** (supported, wrongly declined): mostly short, plainly-
worded queries with real but modest top scores — "What is the law of
trichotomy?" (`bm25_top1=8.1`, `dense_top1=0.65`), "How do you solve
4 - 6x > 22?" (`bm25_top1=10.0`, `dense_top1=0.71`). These aren't
misleading evidence in the way the false accepts are; they're genuinely
weaker matches for genuinely-covered material, the safer of the two error
types by this project's own cost framing (Phase 1G section 11).

Seven test examples is not enough to generalize the false-reject pattern
beyond "some correctly-covered queries score only moderately," and two
false accepts is not enough to rule out sampling noise in an 11%
false-accept rate — both read directionally, not as settled conclusions.

## Limitations

- 36 historical v1 test examples (19 of them unsupported) is small; single-
  percentage-point differences between baselines aren't a stable estimate.
- The near-domain negative set targets concepts *this specific corpus*
  happens to omit — it doesn't generalize to a different or larger corpus
  without re-verifying which concepts are actually absent from it.
- `decision_score` (both the scalar thresholds' raw scores and the
  logistic regression's output) is not a calibrated probability of
  sufficiency — it hasn't been checked for calibration, only for
  separating the two classes at one selected threshold.
- No LLM judge or cross-encoder relevance grader was tried — deliberately
  out of scope for this phase (see below).

## Phase 1H — evidence-support cross-encoder evaluation

Phase 1G's near-domain false accepts share a shape: the retrieved chunk is
lexically and topically close to the query without actually answering it
(`equations-properties-08` names properties it doesn't explain;
`quadratics-formula-01` states a fact about complex roots without deriving
one). A query-passage cross-encoder is a relevance model, not an
entailment model, but relevance and sufficiency correlate often enough
that it's worth measuring directly, on the existing retrieval pipeline's
own top-5 evidence, with no reranking (retrieval order is frozen — this
phase changes one variable, not two) and no LLM judge (that's a
heavier fallback for if this doesn't help, not a baseline to skip ahead
to).

### Model

`cross-encoder/ms-marco-MiniLM-L6-v2`, pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Apache 2.0, 22.7M parameters,
6-layer MiniLM, `max_position_embeddings=512`. Its own config declares
`sbert_ce_default_activation_function: Identity` — `predict()` returns a
raw, unbounded logit, not a probability, and there is no model-provided
"this counts as relevant" cutoff. Loaded via Sentence Transformers'
`CrossEncoder`, already a direct dependency (Phase 1B); no new dependency
was added.

### Evidence and features

For each query, the selected hybrid retriever's existing top-5 chunks
(unchanged retrieval, see above) are scored as (query, chunk.text) pairs.
`EvidenceSupportFeatures` computes `ce_top1`, `ce_top2`, `ce_margin`
(top1 − top2), and `ce_mean_top3` over that fixed rank order; only
`ce_top1` and `ce_margin` are fed into the combined classifier below; the
other two remain available for the scalar baselines. The combined
classifier's 8 features are the 6 Phase 1G retrieval features plus these
2 — no query text, no raw embeddings, no RRF score.

### Freeze order

Model, revision, evidence depth (5), feature set, and the threshold-
selection rule (unchanged from Phase 1G: sweep development scores, keep
the highest-recall threshold reaching ≥0.90 precision) were all fixed
before `algebra_sufficiency_v2.jsonl` was written. The combined evaluator
was run once against development and the historical v1 test with the
holdout file absent (0 rows) to confirm the pipeline was fully functional
before any holdout-influenced number existed. No feature, threshold, or
model choice was changed after the holdout was created or evaluated.

### Fresh holdout: `algebra_sufficiency_v2.jsonl`

60 new queries, `split: "holdout"` throughout, sampled across a written
category plan rather than around the two known Phase 1G failures:

| | count |
|---|---:|
| supported | 28 |
| near-domain unsupported | 22 |
| easy-OOD unsupported | 10 |

Supported queries are spread across equations, inequalities, and
quadratics chunks not exhausted by Phase 1G's own questions on the same
chunks; near-domain queries cover both Phase 1G's documented gaps
(properties, zero-product, factoring, absolute-value equations, rational
root/remainder theorems, systems, matrices, complex roots, graphing,
exponent rules) plus two additional verified gaps (`inverse_property`:
the corpus's properties chunk never names an inverse property;
polynomial-division as long division specifically). Every supported
label's `support_chunk_ids` and every near-domain label's absence were
checked against the actual corpus text, same protocol as Phase 1G — not
generated and trusted. Leakage checks: 60 unique IDs, 60 unique query
strings, zero exact-string overlap with the existing 107, all
`support_chunk_ids` resolve to real chunks, class counts as above. A word-
overlap near-duplicate scan against the old 107 was also run; the two
genuine near-paraphrases it found (a restated zero-product-property
question, a restated discriminant question) were replaced before this
holdout was evaluated — the remaining flagged pairs share only generic
question scaffolding ("what is the ... property"), not the underlying
concept.

### Results

Three approaches, threshold selected on development only, reported
against the historical v1 test and the fresh holdout:

```text
Retrieval-only (Phase 1G's 6-feature logistic, threshold=0.6001, dev precision=0.91, dev recall=0.81)
                       accuracy   precision   recall   f1     false_accept   false_reject
historical v1 test     0.75       0.83        0.59     0.69   0.11           0.41
fresh holdout          0.75       0.81        0.61     0.69   0.12           0.39

Cross-encoder-only (ce_top1 threshold=6.1646, dev precision=0.93, dev recall=0.35)
historical v1 test     0.72       0.89        0.47     0.62   0.05           0.53
fresh holdout          0.65       0.89        0.29     0.43   0.03           0.71

Combined (8-feature logistic, threshold=0.6129, dev precision=0.91, dev recall=0.81)
historical v1 test     0.78       0.85        0.65     0.73   0.11           0.35
fresh holdout          0.72       0.76        0.57     0.65   0.16           0.43
```

Hard negatives (near-domain / easy-OOD rejection rate):

```text
                       historical v1 test          fresh holdout
Retrieval-only         0.85           1.00         0.86           0.90
Cross-encoder-only     0.92           1.00         0.95           1.00
Combined               0.85           1.00         0.82           0.90
```

On the fresh holdout — the primary generalization estimate — neither
cross-encoder approach improves on retrieval-only where it matters most.
Cross-encoder-only has the lowest false-accept rate (0.03) but its recall
collapses to 0.29: it rejects most genuinely-supported queries along with
the unsupported ones, not a usable trade. Combined is worse than
retrieval-only on the fresh holdout on every headline number, including
the metric this phase cares about most: false-accept rate rises from 0.12
to 0.16. The historical v1 test numbers (where combined looks best) do not
generalize; that gap is the result, not a discrepancy to explain away.

### Failure analysis

Representative false accepts from the combined classifier, fresh holdout:

- *"What is the distributive law and where is it used in algebra?"*
  (near-domain) — top evidence is the properties-chapter intro that
  mentions the Commutative Property in passing; `ce_top1=3.99`, a
  relatively high score. This is exactly the pre-registered failure mode:
  the cross-encoder is fooled by the same topical closeness that fools
  retrieval, not corrected by it.
- *"What is the additive identity property?"* (near-domain) — same top
  evidence chunk, but `ce_top1=-1.68`, a low raw logit correctly
  indicating weak relevance. The combined classifier still accepted it:
  retrieval evidence outweighed the cross-encoder signal in this case,
  showing the 8-feature logistic doesn't reliably let a low CE score
  override confident retrieval features.
- *"How do you solve \|2x + 1\| = 7 as an equation rather than an
  inequality?"* and *"How do you solve the system x + y = 5 and
  x - y = 1?"* (both near-domain) — both retrieve topically-adjacent
  chunks (general inequality solving; single-variable equations with
  variables on both sides) that share vocabulary with the query's surface
  form without addressing the actual question (equation-solving for an
  absolute value; a two-variable system). Both score `ce_top1` above 4.0
  — again, relevance-model behavior, not sufficiency-model behavior.
- *"What is the boiling point of water in Kelvin?"* (easy-OOD) — this
  corpus contains a Fahrenheit/Celsius word-problem chunk that discusses
  water's boiling point, so an "easy" out-of-domain query still retrieves
  real, on-topic-sounding evidence (`ce_top1=0.08`, near the low end but
  not low enough). Not every easy-OOD query is easy for every corpus; this
  one only looks obviously out-of-domain until you notice this specific
  corpus happens to discuss the same real-world quantity.

None of these four are the two specific queries Phase 1G's historical
test flagged — the "mentions but doesn't explain" failure mode is a
general property of this corpus and this pipeline, not two isolated
cases, and the cross-encoder does not correct it. The model is not shown
to "understand" insufficiency; it ranks relevance, and relevance keeps
correlating with the same superficial signals retrieval already uses.

### Performance (CPU, descriptive only)

Measured once, locally, single run, no production claim: model load
2.66s; scoring one query's 5-passage evidence set 0.27s; full benchmark
(167 cases across development, historical v1 test, and the fresh holdout,
including retrieval calls) 134s total, ≈0.8s/case — dominated by
retrieval round-trips and dense embedding, not the cross-encoder forward
pass itself.

### Limitations (Phase 1H)

- The fresh holdout is 60 queries; the same small-sample caveat from
  Phase 1G applies to its numbers.
- Only one cross-encoder model was evaluated, by design (see project
  instructions) — this is not a comparison across rerankers, and a
  different model could behave differently.
- The cross-encoder's raw logit is not calibrated or bounded; `ce_top1`
  thresholds found here are specific to this model revision and this
  corpus.
- This phase does not test whether reranking with the cross-encoder
  (changing retrieval order, not just scoring fixed evidence) would help
  — that's a different, still-untested change.

## Phase 1I — LLM evidence-sufficiency judge

Phase 1H's conclusion was that a relevance cross-encoder doesn't fix the
"mentions but doesn't explain" failure — it ranks topical closeness, the
same signal retrieval already uses, not whether the evidence actually
answers the question. Phase 1I asks whether a small LLM, instructed to
read the evidence and judge sufficiency rather than relevance, does
better. This is evaluation only: no answer generation, no web search, no
live routing.

### Provider: OpenAI prepared, never evaluated, replaced with Gemini

The judge was first implemented against OpenAI's Responses API
(`gpt-5.4-nano-2026-03-17`). That implementation was never committed and
never produced a result — no `OPENAI_API_KEY` was available, so the CLI
stopped at the key check every time it was run, before any development,
historical, or v3 judgment existed.

The provider was then switched to Google's Gemini Developer API, for one
reason only: API-key availability/cost (a Gemini free-tier key was
available; the OpenAI integration required paid access that wasn't
authorized). This was **not** a response to any v3 performance signal —
none existed to respond to. The fresh v3 holdout (below) had already been
created before this switch, and its queries and labels were not modified
because of it. The OpenAI code path was deleted rather than kept
alongside Gemini, since it was unfinished and unevaluated, not a working
baseline worth preserving.

The first Gemini model tried, `gemini-2.5-flash`, failed its mandatory
pre-v3 smoke test: the real API returned `404 NOT_FOUND`, "no longer
available to new users," and named `gemini-3.6-flash` as the suggested
replacement. No v3 (or development, or historical) result existed for
`gemini-2.5-flash` — the smoke test uses a synthetic example, not a
benchmark query. That replacement was then explicitly authorized and used
for the one real evaluation reported below; `gemini-2.5-flash` was never
evaluated on anything.

### Model and settings (frozen)

- Model: `gemini-3.6-flash` (Google Gemini Developer API, free tier).
  Gemini does not expose an OpenAI-style pinned dated snapshot for this
  model; the smoke test's returned `model_version` was `gemini-3.6-flash`
  (recorded as returned, not invented). Not chosen for capability — same
  rationale as the original pick: a small, cost-appropriate model for a
  classification-shaped task, not a claim that it's the best possible
  judge.
- Thinking level: `low` (Gemini 3.6 uses named thinking levels, not the
  2.5-era token budget), fixed, not tuned against any result. No thought
  summaries requested or stored — only the final structured judgment.
- API surface: `google-genai`'s `client.models.generate_content()`,
  Structured Outputs (`response_mime_type="application/json"`,
  `response_schema=EvidenceJudgment`), no tools enabled — no web search,
  file search, code execution, or MCP. The judge sees only the evidence
  it's given.
- Prompt identifier: `evidence-sufficiency-v1` — the prompt's actual
  instructions are unchanged since the original OpenAI version (same
  policy, just delivered as `system_instruction`), so the same identifier
  is accurate. One frozen zero-shot prompt
  (`axiom/retrieval/evidence_judge.py`), no few-shot examples from known
  failures, no per-result editing.
- Evidence: the same selected hybrid retriever's top-5 chunks, unchanged
  rank order, same depth as Phase 1G/1H — no reranking.
- Retries: 429 (RESOURCE_EXHAUSTED — free-tier quota) and 5xx are retried
  up to twice with backoff; a 429 that persists past retries stops the
  whole evaluation run rather than continuing to the next dataset (free
  tier quotas are project-specific and not assumed from documentation).

### Judge task and output schema

The prompt is explicit that the question is "does the supplied evidence
contain enough information," not "do you know the answer" — it instructs
the model to ignore outside knowledge, treat a topically-related passage
as not necessarily sufficient, treat a passage that only names a concept
as insufficient, and choose insufficient when uncertain. Evidence text is
marked as untrusted content; instructions appearing inside it are data,
not commands.

Structured output (`EvidenceJudgment`): `verdict` (sufficient/
insufficient), `coverage` (full/partial/none), `supporting_chunk_ids`,
`reason_code` (`complete_support` / `partial_support` /
`related_but_insufficient` / `no_relevant_evidence`), and a short `reason`
string — no hidden chain-of-thought is requested or stored. Two checks
run on every response before it's trusted: `verdict=sufficient` must pair
with `coverage=full` (anything else is rejected as contradictory), and
every `supporting_chunk_ids` entry must be one of the chunk IDs actually
supplied — an invented citation is treated as an evaluation error, not
silently accepted. The decision is the direct verdict
(`sufficient`→USE_KB, `insufficient`→ABSTAIN); there is no confidence
threshold in this first baseline.

### Fresh holdout: `algebra_sufficiency_v3.jsonl`

58 queries, `split: "holdout"`, created only after the judge's original
design (evidence-only policy, structured schema, evidence depth, decision
semantics) was frozen, and before either provider switch described above.
Neither the OpenAI attempt nor the `gemini-2.5-flash` smoke-test failure
produced any v3 judgment; v3 remained unevaluated by any LLM until the
one real `gemini-3.6-flash` run reported below.

| | count |
|---|---:|
| supported | 26 |
| near-domain unsupported | 22 |
| easy-OOD unsupported | 10 |

Sampled by category, not around Phase 1G/1H's two known failures: new
supported queries spread across previously under-used corpus chunks
(e.g. the shepherd-metaphor multiplication/division passage, the
non-worked completing-the-square example, radical-manipulation
identities); near-domain queries cover both previously-documented gaps
and newly-verified ones (closure property, vertex form, piecewise
functions, quadratic inequalities, rational expressions/functions,
logarithms, the fundamental theorem of algebra); easy-OOD spans calculus,
geometry, chemistry, biology, computer science, physics, finance, and
unrelated topics. One deliberate case: a supported query about the
corpus's Fahrenheit/Celsius word-problem chunk, contrasted with an
easy-OOD query about a real-world temperature fact — testing whether the
same accidental content Phase 1H found surprising is handled correctly
when asked on-topic versus off-topic. Every label was checked against the
actual chunk text (not generated and trusted), including confirming
several candidate near-domain terms are genuinely absent from the corpus
by direct search. Leakage checks: 58 unique IDs and query strings, zero
exact-string overlap with the existing 167 (107 + 60) queries; a
word-overlap near-duplicate scan against both prior datasets found three
genuine near-paraphrases (a restated "transitive property of
inequalities" question, a restated Fahrenheit/Celsius question, and a
restated matrix question), each replaced with a distinct angle or a
different verified gap before finalizing.

### Real evaluation: blocked on free-tier quota

`gemini-2.5-flash` failed its mandatory pre-v3 smoke test (`404
NOT_FOUND` — "no longer available to new users", suggesting
`gemini-3.6-flash`). No v3 judgment existed at that point. Per the
governing instructions, `gemini-3.6-flash` was then explicitly authorized
as the sole replacement (no other model was tried), the model/thinking
configuration was updated, and a fresh smoke test — still a synthetic,
non-v3 example — was run before touching v3. It succeeded:

```text
model_version: gemini-3.6-flash
verdict: sufficient   coverage: full   reason_code: complete_support
usage: prompt_tokens=186 candidates_tokens=66 thoughts_tokens=0 total_tokens=252
```

With the design frozen on `gemini-3.6-flash`, the real evaluation
(`--evaluator llm-judge`) was run once, in dataset order: development,
historical v1 test, historical v2 holdout, fresh v3 holdout. It stopped
partway through the **first** dataset:

```text
Gemini free-tier quota exhausted (429) during 'development', after retries.
Completed 11/71 judgments in 'development'.
Stopping without committing an incomplete Phase 1I result.
```

11 of 71 development judgments completed before this project's free-tier
quota was exhausted; the remaining 60 development cases and all of
historical v1 (36), historical v2 (60), and — critically — the fresh v3
holdout (58) were never sent. **No v3 result exists.** Per the governing
instructions ("do not switch models or enable billing... if the free
quota cannot complete the benchmark: STOP and do NOT commit an incomplete
Phase 1I experiment"), no further requests were made, no model swap was
attempted, and v3 was not touched. The 11 completed judgments are not
reported as a metric — 11 development cases is not a meaningful sample
and treating a partial run as a result would be exactly the kind of
fabrication this phase's instructions rule out.

A second attempt (some time later, in the same session) reached only
10/71 development cases before hitting the same condition — confirming
this was not a brief rate-limit spike but a durable quota. AI Studio's
rate-limit page later confirmed this project's actual Gemini 3.6 Flash
free-tier limits: **5 requests/minute, 250K tokens/minute, 20
requests/day** — project-specific figures, not a universal Gemini limit.
A ~225-call four-dataset benchmark cannot complete in one day under a 20
RPD cap, so evaluation execution was made resumable:

- **Pacing** (`axiom/retrieval/judge_execution.py`, `RequestPacer`): real
  request starts are spaced at least 16 seconds apart (≈3.75/minute),
  comfortably under the 5 RPM limit, without changing what each request
  contains or how it's judged.
- **Cache** (`JudgeCache`, `.data/evaluation-cache/`, git-ignored): every
  successfully-validated judgment is persisted keyed by a hash of
  provider, model, prompt version, thinking level, the query, the
  evidence chunk IDs and their actual text, and the output schema's own
  fingerprint. Any change to model, prompt, thinking level, retrieval, or
  schema silently invalidates old entries — a stale judgment is never
  reused. Failures, invalid responses, and quota-limited attempts are
  never cached. On startup, the evaluator skips any case whose exact
  identity is already cached and only pays the pacing cost for the rest,
  so the benchmark can resume across multiple days without resending
  already-successful requests.
- No successful judgments from the two earlier (pre-cache) attempts could
  be recovered — the cache didn't exist yet when those 11 and 10 calls
  were made, so those responses were never persisted. The cache starts
  empty; the next run pays for all ~225 calls once, and any run
  interrupted after that resumes from whatever was cached.

`axiom/retrieval/evidence_judge.py` (the judge itself — model, prompt,
schema, decision semantics) is unchanged and unit-tested against a fake
client; only execution reliability was added. Re-running
`uv run python -m axiom.retrieval.evaluate_sufficiency --evaluator
llm-judge` once quota resets requires no code change, and will resume
rather than restart.

### Execution-order change: v3 first, historical deferred

The original plan (Phase 1H's precedent) was to evaluate development and
both historical sets before the fresh holdout. After observing this
project's 20-requests/day free-tier constraint — but before any v3
judgment existed — that order was changed: `--evaluator llm-judge-v3`
evaluates only the fresh v3 holdout against the frozen Gemini judge,
skipping the 71+36+60=167 development/historical Gemini calls entirely.

The reasoning: this judge is a frozen zero-shot evaluator with no
training and no threshold fitting (unlike the retrieval-only logistic
baseline, which *is* fit on development, but that fitting is entirely
local — no Gemini calls). Nothing about running development or the
historical sets through Gemini would change the judge's configuration,
so they aren't required to determine it. Running all 167 of them first
would cost roughly 9 additional free-tier quota days without improving
the validity of v3 — the one generalization estimate this phase cares
about. Development/historical Gemini evaluation is deferred, not
skipped permanently: `--evaluator llm-judge` (the original, full
four-dataset path) remains available and may be run later if its
comparison value justifies the quota, but is not required for the Phase
1I decision.

This is an execution-order change, not a benchmark optimization: it was
made with zero v3 judgments in hand, v3's queries and labels were not
touched, and the retrieval-only baseline used for comparison is the same
frozen Phase 1G model regardless of which order the LLM judge runs in.

A further attempt at `--evaluator llm-judge-v3 --provider gemini`, after
the user-reported quota reset, reached only 1/58 before hitting the same
429 condition again almost immediately — confirming this was the daily
quota, not a one-off rate-limit blip that a brief wait would clear. One
valid Gemini v3 judgment exists in the cache; it is kept, but — being a
single case — is not reported as a metric.

### Provider switch: Gemini exhausted, Meta authorized

Before any aggregate v3 result existed for any provider, Gemini's
free-tier was replaced with a second provider — Google's Gemini
Developer API was not abandoned by choice; the actual free quota made a
58-call run impractical to complete in reasonable time. Meta's Model API
(`muse-spark-1.3`, reached through the official `openai` Python SDK
pointed at `https://api.meta.ai/v1` — Meta's documented integration
method, not a new abstraction) was explicitly authorized as a second,
separate provider:

- Same frozen design throughout: prompt `evidence-sufficiency-v1`
  (identical instructions, byte-for-byte), `EvidenceJudgment` schema and
  validation, top-5 hybrid evidence, decision semantics
  (`sufficient`→USE_KB). Only the provider-specific settings differ:
  model `muse-spark-1.3`, reasoning `effort=low` (Meta's equivalent of
  Gemini's `thinking_level=low`).
- `MetaEvidenceJudge` (`axiom/retrieval/evidence_judge.py`) refuses to
  send the API key unless `MODEL_API_BASE_URL` matches
  `https://api.meta.ai/v1` exactly (`MetaCredentialError` otherwise). A
  429 (rate limit) is retried with backoff; a 402 (billing/insufficient
  credit) is not retryable and is treated as a stop-the-run condition,
  same as a persistent quota exhaustion.
- `judge_execution.py::cache_key()` was extended with an explicit
  `provider` field (previously hardcoded to `"gemini"`); Gemini's one
  existing cache entry was verified to still resolve correctly under the
  new signature before any Meta call was made.
- A mandatory pre-v3 smoke test (synthetic, non-v3 example) was run
  first and succeeded: `model_version=muse-spark-1.3`,
  `verdict=sufficient coverage=full`, usage
  `prompt_tokens=341 candidates_tokens=392 thoughts_tokens=329
  total_tokens=733`, latency 5.14s. Per instruction, the run then paused
  for manual confirmation of Meta billing/credit coverage before any v3
  case was sent — no v3 judgment existed until that confirmation.

### Real v3 result: Meta `muse-spark-1.3`, 58/58

Run once, paced (≥16s between request starts), after the freeze above —
no model, prompt, schema, thinking setting, retrieval, or v3 question/
label was changed during or after the run.

```text
                        accuracy  precision  recall  F1    false_accept  false_reject
Retrieval-only (1G)     0.84      0.81       0.85    0.83  0.16          0.15
Meta muse-spark-1.3     0.98      1.00       0.96    0.98  0.00          0.04
```

Confusion matrices (n=58): retrieval-only `tp=22 fp=5 tn=27 fn=4`; Meta
`tp=25 fp=0 tn=32 fn=1`. Hard negatives: retrieval-only near-domain 0.77,
easy-OOD 1.00; Meta near-domain 1.00, easy-OOD 1.00. Zero evaluation
errors (58/58 valid structured judgments, zero invalid/contradictory
responses).

Reason-code distribution, cross-tabulated with whether the verdict
matched the true label:

```text
complete_support           25 correct, 0 wrong
no_relevant_evidence       24 correct, 0 wrong
related_but_insufficient    7 correct, 0 wrong
partial_support             1 correct, 1 wrong
```

The `related_but_insufficient` row is the number this phase exists to
measure: all 7 near-domain cases the judge placed in that category were
correctly rejected — this is exactly the "topically related, but doesn't
contain enough to answer" failure class Phase 1H's cross-encoder could
not reliably separate from genuine sufficiency. Representative examples
(all correctly judged insufficient):

- *"What is the closure property of real numbers?"* — "mentions
  properties of real numbers (Commutative, Associative, Distributive,
  Identity) but does not state or define the closure property."
- *"How do you factor a quadratic by grouping?"* — "describes completing
  the square, the quadratic formula, and solving linear equations, but
  none of the chunks describe the grouping method... topically related
  to quadratics but lack the specific grouping steps."
- *"What is the multiplicative identity property?"* — "no chunk defines
  the multiplicative identity property... the closest text only names
  the Identity Property in passing."

**The one error** (the only misclassification across all 58 cases, a
false reject): *"What does it mean for a quadratic's solutions to be
'conjugate imaginary' rather than real?"* (`quadratics-formula-01`
provides the discriminant condition for when solutions are "conjugate
imaginary"). The judge's `partial_support` reasoning: "Evidence states
the discriminant condition for conjugate imaginary (Δ<0) vs real (Δ>0),
but does not define what 'conjugate imaginary' means or how it differs
from real solutions." A defensible, conservative reading — the chunk
does use the term without defining it — and it erred toward the safe
direction (reject, not accept).

Zero false accepts occurred — no near-domain or easy-OOD unsupported
query was ever judged sufficient.

**Usage**: 58 requests, 0 errors. `prompt_tokens=112953
candidates_tokens=34960 thoughts_tokens=29641 total_tokens=147913`.
Covered by this project's Meta Model API free credits (confirmed via the
Meta Billing dashboard: "$142.94 Free credits in use", "₹0.00
outstanding" before the run) — $0.00 out-of-pocket for this run; no
per-token dollar rate is claimed or computed, since none was
independently verified.

**Latency**: mean 5.03s, median 4.55s, p95 8.73s, total wall-clock 292.0s
for all 58 requests (descriptive, single run, not a production claim).

### Limitations (Phase 1I)

- 58 v3 cases (and 36/60 on the historical sets, never run against any
  LLM judge) is a modest sample; a single-digit swing in false accepts or
  rejects would change the headline rates visibly. One error in 58 cases
  is not enough to characterize the judge's full failure distribution.
- Two LLM providers were evaluated, but not by original design — Meta
  was added only because Gemini's free tier made completing the
  benchmark impractical, not as a deliberate comparison. No OpenAI result
  exists; no comparison across three providers is implied or claimed.
- `reason_code` is diagnostic only; it is never used for the routing
  decision, which stays the explicit `verdict`.
- Retry/backoff (bounded, `max_retries=2`) covers transient errors
  (429/5xx for Gemini, 429 for Meta); it cannot help once a persistent
  quota or billing condition is hit, by either provider.
- The historical v1/v2 sets were never run against the LLM judge (either
  provider) — deferred to conserve quota, not because they were judged
  unnecessary; comparisons in this document are limited to v3.
- Meta's exact per-token pricing was not independently verified; the
  $0.00 figure reflects this run being covered by existing free credits,
  not a verified permanently-free tier.

## Related work, briefly

This two-step shape — retrieve, then assess whether the retrieved
evidence is actually sufficient before deciding what to do next — is the
same shape retrieval-evaluator approaches like Self-RAG and Corrective RAG
use. Nothing here implements either of those; this phase measures whether
the simplest version of that idea (retrieval scores alone, no learned or
LLM-based relevance judge) works well enough to be worth building on.

## Deferred

Phase 1H evaluated a cross-encoder relevance judge; it did not materially
help and is not wired into any live path. Phase 1I implements an LLM
evidence judge but has not yet produced a real result (no API key
authorized this session — see above). Still unimplemented: web search /
MCP fallback, query rewriting, cross-encoder reranking, LangGraph
orchestration, guardrails, human-in-the-loop feedback, SymPy verification,
and adaptive tutoring. All three phases only produce and evaluate the
routing decision in isolation — nothing here is wired into a live request
path.
