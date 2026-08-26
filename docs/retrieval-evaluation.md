# Retrieval evaluation (Phase 1A)

This documents the first measured retrieval baseline, before any vector
store or embedding model exists. The goal was to answer a narrower question
than "does RAG work": does lexical retrieval over a small, real curriculum
corpus find the right chunk, and does its score tell us anything about when
the knowledge base doesn't cover a question. Both are prerequisites for the
KB-first-with-web-fallback design in `docs/architecture.md` — there's no
point choosing a confidence threshold before checking whether the score
distributions it would threshold are actually separable.

## Corpus

Seven Wikibooks: Algebra pages, 40 chunks total, covering equations and
their properties, inequalities, and quadratic equations. See
`data/README.md` for the source list and the topics deliberately left out.

## Labeling process

Every one of the 50 queries in `data/evaluation/algebra_retrieval.jsonl` was
written after reading the actual chunk text it's labeled against, dumped
from a real ingestion run — not generated first and assumed correct. A
handful of queries reuse near-duplicate chunks (e.g. the corpus contains two
close variants of the "variable in the denominator" method, from repeated
content in the source page itself); those are labeled with both chunk IDs
rather than picking one arbitrarily. 8 of the 50 queries are out-of-scope
(calculus, geometry, probability, and unrelated topics), with no labeled
relevant chunks, to support the confidence analysis below.

## Baseline

BM25 (`rank_bm25.BM25Okapi`) over whitespace/punctuation-insensitive
lowercased tokens (`axiom.retrieval.bm25.tokenize`). No stemming, no stop
word removal, no query expansion. This is deliberately the simplest thing
that could plausibly work — the point of Phase 1A is a real number to
compare a future retriever against, not a tuned baseline.

## Results

Measured by `uv run python -m axiom.retrieval.evaluate` against the
committed corpus and evaluation set:

```text
Hit@1          0.52
Hit@3          0.79
Hit@5          0.86
Recall@5       0.83
MRR            0.66
```

nDCG was left out: relevance labels here are binary, so it wouldn't say
anything Recall@5 and MRR don't already say between them.

Reading these plainly: BM25 usually finds a relevant chunk in the top 5
(86% of the time) but is a coin flip on getting it into first place. The
gap between Hit@1 (0.52) and Hit@3 (0.79) is the most informative number
here — it suggests a lot of the near-misses are lexical (a paraphrase or a
conceptual question that doesn't share the corpus's exact wording), which is
exactly the class of query a future dense retriever should help with.

## Confidence investigation

Top-1 BM25 score, in-scope vs. out-of-scope queries:

```text
in-scope      n=42  min=5.75  median=10.11  max=22.13  mean=11.22
out-of-scope  n=8   min=3.67  median=6.15   max=9.68   mean=6.26
```

These overlap. Out-of-scope queries generally score lower, but not reliably
enough to threshold on: at a cutoff of 8.0, 8 of 42 in-scope queries (19%)
would fall below it and be wrongly treated as "the KB doesn't cover this,"
while 1 of 8 out-of-scope queries (12.5%) would clear it and be wrongly
treated as covered. No cutoff in the overlap region (roughly 6 to 10) does
meaningfully better than that trade-off.

**Conclusion: raw BM25 top-1 score is not, by itself, a sound basis for the
KB-sufficiency threshold the architecture calls for.** This isn't a reason
to abandon a threshold-based fallback decision — it's a reason not to pick
one yet. `threshold = 0.7` (or any other specific number) would be exactly
the kind of arbitrary choice this investigation exists to avoid.

## Limitations

- **LaTeX leakage.** Formulas are extracted from MediaWiki's `alttext`
  attribute and kept close to their explanation (see
  `axiom/knowledge/chunking.py`), but the result is raw LaTeX commands
  (`\frac`, `\sqrt`, `\cdot`) sitting in otherwise plain text, not rendered
  math. This makes some chunks (e.g. `equations-properties-05`, a table of
  radical identities) read poorly and probably hurts their retrievability
  for queries phrased in plain English.
- **Duplicate content.** The inequalities source pages repeat several
  sections near-verbatim (`inequalities-basics-03` and `-08`, `-05` and
  `-09`). This is a property of the source, not an ingestion bug, but it
  means the corpus has less unique content than its chunk count suggests.
- **Thin chunks.** A few chunks are stubs or list fragments with little
  standalone information (`quadratics-formula-02` is a placeholder the
  source itself never filled in; `inequalities-basics-07` lost its
  definition-list body during extraction). They weren't targeted by any
  eval query, but they sit in the corpus as low-value noise a future
  retriever also has to rank against.
- **Small evaluation set.** 50 queries is enough to see a real, honest
  separation problem in the score distributions; it is not enough to treat
  any of the metrics above as a stable estimate. Precision on differences
  smaller than a few percentage points isn't warranted.

## What Phase 1B should test next

Given the above, the most useful next experiment is whether a dense
retriever closes the Hit@1/Hit@3 gap on paraphrased and conceptual queries
specifically (the query_type field in the evaluation set already supports
slicing metrics by type), and whether embedding similarity separates
in-scope from out-of-scope queries more cleanly than BM25's score does. That
argues for introducing embeddings and Qdrant next, evaluated against this
same query set before anything else changes, rather than a hybrid or
reranking step that would be premature to justify.
