# Retrieval evaluation

This documents the retrieval baselines measured so far, before any vector
store, LLM, or answer-generation exists. The goal is to answer a narrower
question than "does RAG work": does a given retriever find the right chunk
over a small, real curriculum corpus, and does its score tell us anything
about when the knowledge base doesn't cover a question. Both are
prerequisites for the KB-first-with-web-fallback design in
`docs/architecture.md` — there's no point choosing a confidence threshold
before checking whether the score distributions it would threshold are
actually separable.

## Corpus

Seven Wikibooks: Algebra pages, 40 chunks total, covering equations and
their properties, inequalities, and quadratic equations. See
`data/README.md` for the source list and the topics deliberately left out.
Phase 1B reuses this corpus and the Phase 1A evaluation set unchanged — no
chunk, query, or label was altered to make either retriever look better.

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

## Phase 1A — BM25 baseline

BM25 (`rank_bm25.BM25Okapi`) over whitespace/punctuation-insensitive
lowercased tokens (`axiom.retrieval.bm25.tokenize`). No stemming, no stop
word removal, no query expansion. This is deliberately the simplest thing
that could plausibly work — the point was a real number to compare a future
retriever against, not a tuned baseline.

```text
Hit@1          0.52
Hit@3          0.79
Hit@5          0.86
Recall@5       0.83
MRR            0.66
```

nDCG was left out because Hit@k, Recall@5, and MRR were sufficient to
characterize this baseline — not because binary relevance makes nDCG
uninformative; several queries here do have more than one labeled relevant
chunk, and nDCG would still distinguish ranking those higher from ranking
them lower. It's a scope decision, revisit if a future comparison needs
that distinction.

Reading these plainly: BM25 usually finds a relevant chunk in the top 5
(86% of the time) but is a coin flip on getting it into first place. The
gap between Hit@1 (0.52) and Hit@3 (0.79) is the most informative number
here — it suggests a lot of the near-misses are lexical (a paraphrase or a
conceptual question that doesn't share the corpus's exact wording).

Top-1 BM25 score, in-scope vs. out-of-scope queries:

```text
in-scope      n=42  min=5.75  median=10.11  max=22.13  mean=11.22
out-of-scope  n=8   min=3.67  median=6.15   max=9.68   mean=6.26
```

These overlap. Out-of-scope queries generally score lower, but not reliably
enough to threshold on: at a cutoff of 8.0, 8 of 42 in-scope queries (19%)
would fall below it and be wrongly treated as "the KB doesn't cover this,"
while 1 of 8 out-of-scope queries (12.5%) would clear it and be wrongly
treated as covered. Raw BM25 top-1 score is not, by itself, a sound basis
for a KB-sufficiency threshold.

## Phase 1B — dense retrieval experiment

### Model

[`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base),
pinned to revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. Facts below
are from the model's own `config.json` and sentence-transformers config, not
from marketing copy on the model card:

- License: Apache 2.0
- Embedding dimension: 768
- Max sequence length: 8192 tokens
- Similarity function: cosine (`similarity_fn_name: cosine` in the model's
  `config_sentence_transformers.json`); embeddings are encoded with
  `normalize_embeddings=True`, so cosine similarity reduces to a dot product
- Query/document encoding: symmetric. The model's sentence-transformers
  config declares no prompt (`default_prompt_name: null`), unlike
  asymmetric models such as E5 or BGE that expect a `"query: "` /
  `"passage: "` prefix, so `axiom.retrieval.dense.SentenceTransformerEmbedder`
  encodes queries and chunks the same way.

This is the first dense baseline tried, not a claim that it's the best
available embedding model — no other model was evaluated against this
benchmark.

### Evaluation method

`DenseRetriever` (`axiom/retrieval/dense.py`) implements the same
`Retriever` protocol as `BM25Retriever`: it embeds all 40 chunks once,
embeds each query, ranks chunks by dot product, and returns
`RetrievalResult` objects the evaluation harness already knows how to score.
The embedding step is injected as a small `Embedder` protocol so unit tests
run against a fake embedder — `pytest` never downloads the model or touches
the network.

```bash
uv run python -m axiom.retrieval.evaluate --retriever dense
uv run python -m axiom.retrieval.evaluate --retriever compare
```

The model downloads from Hugging Face on first use (cached under
`~/.cache/huggingface`, outside the repository) and is otherwise
reproducible from the model ID and pinned revision above.

### Results

```text
Metric          BM25   Dense   Delta
Hit@1           0.52    0.69   +0.17
Hit@3           0.79    0.86   +0.07
Hit@5           0.86    0.88   +0.02
Recall@5        0.83    0.87   +0.04
MRR             0.66    0.78   +0.12
```

Dense wins on every metric, and the win is concentrated exactly where BM25
was weakest: Hit@1 improves by 17 points while Hit@5 — where BM25 was
already reasonably strong — only improves by 2. That's the shape of result
the Phase 1A writeup predicted (Hit@1 was the main weak point), not an
across-the-board improvement.

### Query-type breakdown

```text
                n   BM25 Hit@1 / MRR   Dense Hit@1 / MRR
application    11   0.73 / 0.83        0.55 / 0.66
conceptual      7   0.29 / 0.50        0.71 / 0.79
confusable      2   0.50 / 0.50        0.50 / 0.67
definition      8   0.50 / 0.69        0.88 / 0.94
formula         4   0.50 / 0.58        0.50 / 0.62
method          5   0.40 / 0.50        0.80 / 0.80
paraphrase      3   0.67 / 0.75        1.00 / 1.00
terminology     2   0.50 / 0.75        0.50 / 0.75
```

Dense clearly helps conceptual, definition, method, and paraphrase queries —
the categories where a query is unlikely to share exact vocabulary with the
chunk that answers it. It does not help formula or terminology queries
(both flat), and it is worse on `application` queries (worked problems like
`"solve 2x^2 + 24x + 23 = 0 by completing the square"`), where matching the
literal numbers and operators in the problem statement is exactly what a
lexical method is good at and semantic similarity doesn't specifically
reward. `confusable` and `terminology` groups are too small (n=2) to read
into.

### Paired top-1 outcome (in-scope queries, n=42)

```text
both correct     20
bm25 only          2
dense only          9
both wrong        11
```

Dense fixes 9 queries BM25 got wrong at rank 1 and breaks 2 that BM25 had
right — a clearly favorable trade, but 11 queries neither retriever gets
right, so dense is not a solved problem here. The two cases dense broke are
informative: `alg-ret-004` ("Solve 3x + 1 = 4 for x") and `alg-ret-016`
("If a = b and 1 + a = 3, can I replace a with b?") both have exact-token
overlap with their correct chunk that BM25 rewards directly and dense's
semantic similarity apparently doesn't weight as heavily. The 9 dense-only
fixes are mostly paraphrases and conceptual questions with no literal
overlap with the source text, e.g. `alg-ret-023` ("What are the properties
of inequalities?") — BM25 matched a chunk about equality properties on
lexical overlap with "properties"; dense correctly matched the inequality
properties chunk.

### Confidence investigation

Top-1 dense score, in-scope vs. out-of-scope queries:

```text
in-scope      n=42  min=0.61  median=0.78  max=0.87  mean=0.77
out-of-scope  n=8   min=0.52  median=0.56  max=0.67  mean=0.58
```

These also overlap — out-of-scope max (0.67) is above in-scope min (0.61).
On raw score alone this looks like a tighter overlap than BM25's, but the
two scales aren't comparable (dense scores are cosine similarities bounded
in [-1, 1]; BM25 scores are unbounded). Measured as a fraction of each
retriever's own in-scope range, the overlap is actually similar: BM25's
overlap band (5.75–9.68) is 24% of its in-scope range (5.75–22.13); dense's
overlap band (0.61–0.67) is 23% of its in-scope range (0.61–0.87).
**Dense similarity does not appear to separate in-scope from out-of-scope
queries meaningfully better than BM25's score did**, once the comparison
accounts for the different scales. No threshold is set — the out-of-scope
sample is 8 queries, too small to support a confident threshold decision
either way, and this is a reason for caution about that number specifically,
not just a general disclaimer.

### Performance

Local, single-machine, CPU-only measurements (no GPU, no `flash_attn`,
model already downloaded) — descriptive, not a production latency claim:

- Corpus embedding (40 chunks, one batched `encode()` call): BM25 0.00s
  (tokenization only); dense 300.57s
- Query retrieval (50 queries, one `encode()` call per query, no batching
  across queries): BM25 0.01s total (0.2ms/query); dense 9.49s total
  (189.8ms/query)

Dense corpus embedding is the striking number here — five minutes for 40
short text chunks in a single batched call. Queries, embedded individually,
average a much more reasonable 190ms each; chunk texts run up to roughly
2000 characters versus a query's one sentence, and ModernBERT's attention
cost does not scale linearly with sequence length, so this is plausibly a
real effect of chunk length rather than a batching bug — but it was not
isolated and confirmed against per-chunk length. Both numbers are almost
certainly inflated by the missing `flash_attn` optional dependency (the
model card recommends it) on this CPU-only machine. This doesn't affect the
retrieval-quality comparison above, but it is a real constraint on scaling
the corpus size or embedding it synchronously in Phase 1's current
in-memory design — five minutes to (re-)embed 40 chunks does not scale to a
real curriculum KB without batching, caching, or a GPU.

## Limitations

- **LaTeX leakage.** Formulas are extracted from MediaWiki's `alttext`
  attribute and kept close to their explanation (see
  `axiom/knowledge/chunking.py`), but the result is raw LaTeX commands
  (`\frac`, `\sqrt`, `\cdot`) sitting in otherwise plain text, not rendered
  math. This affects both retrievers equally but was not re-measured
  per-retriever.
- **Duplicate content.** The inequalities source pages repeat several
  sections near-verbatim (`inequalities-basics-03` and `-08`, `-05` and
  `-09`). This is a property of the source, not an ingestion bug, but it
  means the corpus has less unique content than its chunk count suggests.
- **Thin chunks.** A few chunks are stubs or list fragments with little
  standalone information (`quadratics-formula-02`, `inequalities-basics-07`).
  They weren't targeted by any eval query but sit in the corpus as noise
  both retrievers rank against.
- **Small evaluation set.** 50 queries, 8 of them out-of-scope, is enough to
  see a real, honest pattern in the results above; it is not enough to
  treat any single percentage-point difference as a stable estimate, and is
  specifically too small to justify a KB-sufficiency threshold from either
  retriever's score distribution.
- **One model, one run.** No other embedding model, no query prefixing
  experiment, and no repeated runs for variance were tried — this is a
  first baseline, not a tuned or validated choice.

## Next steps

Phase 1B shows dense retrieval measurably outperforms BM25 on this corpus
and query set, concentrated on paraphrase/conceptual/definition queries,
while being no better (and on `application` queries, slightly worse) where
literal term overlap already works. It does not show that dense similarity
solves KB-sufficiency routing — that overlap problem persists proportionally
across both retrievers. The most direct next experiment is BM25+dense hybrid
retrieval (combine both rankings rather than replace one with the other,
since their failure cases are largely disjoint per the paired analysis
above) evaluated against this same query set, before considering Qdrant,
reranking, or a second embedding model.
