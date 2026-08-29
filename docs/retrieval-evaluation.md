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
Phases 1B, 1C, and 1D reuse this corpus and the Phase 1A evaluation set
unchanged — no chunk, query, or label was altered to make any retriever
look better.

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

## Phase 1C — hybrid retrieval experiment

Phase 1B's paired failure analysis found BM25 and dense mostly fail on
*different* queries (2 bm25-only successes, 9 dense-only successes, 20 both
correct). That complementary pattern is the reason to try combining rankings
rather than picking one retriever.

### Method

[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
(Cormack et al., 2009): each component retriever returns its top 10
candidates (`CANDIDATE_DEPTH`, a quarter of the 40-chunk corpus); a chunk's
fused score is `sum(1 / (k + rank))` over every list it appears in, using
`k=60` — the constant from the original paper, not tuned against this
benchmark. Rank position is fused, not raw score, because BM25's unbounded
score and dense's cosine similarity aren't on comparable scales.
`HybridRetriever` (`axiom/retrieval/hybrid.py`) wraps an already-constructed
`BM25Retriever` and `DenseRetriever` rather than building its own index.
Ties in fused score break on chunk ID, so ranking doesn't depend on
insertion order.

```bash
uv run python -m axiom.retrieval.evaluate --retriever hybrid
uv run python -m axiom.retrieval.evaluate --retriever compare
```

### Results

```text
Metric        BM25   Dense   Hybrid   Hyb-Dns   Hyb-BM25
Hit@1         0.52   0.69    0.74     +0.05     +0.21
Hit@3         0.79   0.86    0.88     +0.02     +0.10
Hit@5         0.86   0.88    0.88     +0.00     +0.02
Recall@5      0.83   0.87    0.88     +0.01     +0.05
MRR           0.66   0.78    0.81     +0.03     +0.15
```

Hybrid beats dense — the stronger of the two baselines — on every metric,
not just BM25. The gain is concentrated at Hit@1/MRR, the metrics dense
already improved on over BM25, while Hit@5/Recall@5 are unchanged or
slightly better: fusion isn't just recovering BM25's ranking on cases dense
already got into the top 5, it's improving which chunk lands first.

### Query-type breakdown

```text
                n   BM25 Hit@1   Dense Hit@1   Hybrid Hit@1
application    11   0.73          0.55          0.82
conceptual      7   0.29          0.71          0.71
confusable      2   0.50          0.50          0.50
definition      8   0.50          0.88          0.88
formula         4   0.50          0.50          0.50
method          5   0.40          0.80          0.80
paraphrase      3   0.67          1.00          0.67
terminology     2   0.50          0.50          0.50
```

The notable change is `application`: Phase 1B found dense regressed below
BM25 here (worked-problem queries where literal token overlap matters).
Hybrid recovers it, and exceeds BM25 (0.82 vs. 0.73) — rank fusion gives
BM25's exact-match ranking enough weight to win when dense's semantic
ranking alone would have lost. `conceptual`, `definition`, and `method`
hold at dense's improved level. `paraphrase` (n=3) drops from 1.00 to 0.67 —
one query out of three; too small a group to read as a real regression
rather than noise.

### Paired top-1 outcome (in-scope queries, n=42)

Relative to dense: hybrid fixes 4 of dense's failures and breaks 2 of
dense's successes; 3 of the 4 fixes are cases where BM25 also failed, so
fusion is finding chunks neither component ranked first alone, not just
picking whichever component was right.

Against Phase 1B's specific complementary cases: hybrid recovers 1 of the 2
bm25-only successes (`alg-ret-004`, "Solve 3x + 1 = 4 for x") but not the
other (`alg-ret-016`), and preserves 7 of the 9 dense-only successes,
losing `alg-ret-013` ("Is the square root of a number always positive?")
and `alg-ret-032` ("using absolute value to check if a measurement is
within range") — both cases where BM25's competing candidate outranked the
correct one enough in the fused score to win. Hybrid is not a strict
superset of what either component gets right.

### Confidence investigation

```text
Top-1 score, in-scope      n=42  min=0.03  median=0.03  max=0.03  mean=0.03
Top-1 score, out-of-scope  n=8   min=0.03  median=0.03  max=0.03  mean=0.03
```

RRF's fused score has almost no dynamic range here — most top-1 scores
round to the same two decimal places regardless of scope, because with
`k=60` the difference between rank 1 and rank 2 in either list is tiny
(`1/61` vs. `1/62`), and a top-1 hit is usually rank 1 or 2 in both
component lists. This makes RRF's own score a *worse* candidate for a
KB-sufficiency signal than either BM25's or dense's raw score, not a
better one — it's a ranking score by design, not a calibrated estimate of
relevance. As with Phase 1A/1B, no threshold is set, and the 8-query
out-of-scope sample remains too small to support one either way.

### Performance

CPU-only, descriptive only: hybrid's own construction is free (it embeds
nothing; it reuses the bm25/dense retrievers built for the comparison run).
Per-query hybrid retrieval (6.43s / 50 queries, 128.6ms/query) is
essentially dense's query time (6.39s, 127.8ms/query) plus BM25's
negligible cost — fusing 20 candidates per query is not the bottleneck.

### Limitations specific to Phase 1C

- `k=60` and depth=10 were fixed from convention/corpus size, not searched;
  a different constant or depth was not tried.
- The `paraphrase` regression (1 of 3 queries) and the two lost dense-only
  successes are each single-digit-count findings on a 50-query benchmark —
  suggestive, not conclusive.
- Only two components were fused. No reranking, no weighting one retriever
  above the other, no third signal.

## Phase 1D — Qdrant-backed persistent dense retrieval

Phase 1B/1C's `DenseRetriever` re-embeds all 40 chunks every process start
(~200-300s on this CPU-only machine). This phase asks a narrower question
than "is Qdrant a good vector store": does moving dense storage/search to
Qdrant preserve the retrieval quality already measured, while embedding the
corpus once instead of every run? Only the dense retriever's storage/search
changed — `BM25Retriever` and the RRF fusion in `hybrid.py` are untouched,
so a `QdrantDenseRetriever` result is directly comparable to Phase 1B/1C's.

### Deployment

`QdrantClient(path=...)` local persistent mode, not a server. The corpus is
40 chunks; local mode gives real on-disk persistence without adding Docker
as a project dependency, and this phase only needed collection creation,
upsert, cosine search, and payload retrieval — all of which local mode
supports directly. Storage lives at `.data/qdrant/` (git-ignored by the
existing `/.data/` rule; override with `AXIOM_QDRANT_PATH`). Docker is
available on this machine but wasn't used — nothing here needed it.

### Collection

`axiom_algebra`: one unnamed dense vector, size 768, cosine distance —
matching `Alibaba-NLP/gte-modernbert-base` (same model, same pinned
revision as Phase 1B; no embedding change). `ensure_collection`
(`axiom/retrieval/qdrant_store.py`) validates an existing collection's
vector config rather than silently reusing or deleting a mismatched one; a
`--recreate` flag on the indexing command is required to rebuild it.

Chunk IDs (e.g. `equations-properties-02`) aren't valid Qdrant point IDs, so
each point's ID is a UUID5 deterministically derived from the chunk ID
(`point_id_for_chunk`) — re-indexing the same chunk always maps to the same
point. The payload is the chunk's own `model_dump()`, so `chunk_id` and
every other provenance field (source URL, license, section) round-trip
through Qdrant without a separate hand-maintained schema.

### Indexing

```bash
uv run python -m axiom.retrieval.index_qdrant
uv run python -m axiom.retrieval.index_qdrant --recreate
```

Deterministic point IDs mean this is naturally idempotent: indexing the
committed corpus twice produced 40 points both times, not 80 (`Collection
created: False` on the second run, confirmed by an actual second run, not
inferred).

### Results

```text
Retriever         Hit@1   Hit@3   Hit@5  Recall@5   MRR
BM25               0.52    0.79    0.86      0.83  0.66
Dense local        0.69    0.86    0.88      0.87  0.78
Dense Qdrant       0.69    0.86    0.88      0.87  0.78
Hybrid local       0.74    0.88    0.88      0.88  0.81
Hybrid Qdrant      0.74    0.88    0.88      0.88  0.81
```

Identical to five significant figures at this precision. Ranking agreement,
not just aggregate metrics: dense local vs. Qdrant dense agree on the top-1
result for all 50 queries, and on the top-5 *set* for all 50 (mean Jaccard
1.00) — though for 2 of those 50 the order within an unchanged set differs
at rank 3+. Hybrid local vs. Qdrant hybrid also agree on top-1 for all 50
queries; mean top-5 Jaccard is 0.99, not exactly 1.00, because RRF's
rank-sensitivity turns one of those ordering differences into an actual
top-5 set change for a different query. See "Ranking difference,
investigated" below for what was actually confirmed about the cause. None
of this changed any query's top-1 outcome or any metric above.

### Ranking difference, investigated

Qdrant's local mode does not use HNSW or another approximate index — its
search computes cosine similarity over every stored vector and sorts
exactly (`qdrant_client.local.local_collection`, confirmed by reading the
installed package: a full `np.argsort` over the complete score array, no
approximate graph traversal). The earlier draft of this document attributed
the ranking difference to HNSW; that was wrong for this configuration and
has been corrected.

Reproduced directly with the real corpus and model: local `DenseRetriever`
and `QdrantDenseRetriever` disagree on ranking *order* — not on which
chunks are retrieved — for 2 of 50 queries:

- `alg-ret-034` ("graph x > 3 or x < 2 on a number line"): local scores
  `inequalities-basics-06` and `-10` as an exact tie, 0.712402 both: Qdrant
  scores them 0.712274 and 0.712447, breaking the tie the other way.
- `alg-ret-039` ("What does the discriminant tell me about the roots?"):
  local and Qdrant both place `quadratics-completing-square-01`/`-02` at
  4th/5th but disagree on which is which; the gap is under 0.0005 either
  way.

Propagated through RRF, which is sensitive to rank rather than raw score,
this becomes a genuine top-5 *set* change for a third query, `alg-ret-016`
("If a = b and 1 + a = 3, can I replace a with b?"): the fifth-ranked chunk
in the fused result is `inequalities-basics-06` locally and
`quadratics-completing-square-01` via Qdrant.

A concrete, evidence-based contributing factor: `SentenceTransformerEmbedder.encode`
returns `float16` embeddings (confirmed), and Qdrant's local-mode cosine
similarity re-normalizes both query and stored vectors at search time
(`qdrant_client.local.distances.cosine_similarity`), even though
`DenseRetriever` already normalizes during encoding and takes a raw dot
product instead of repeating that step. The two paths can therefore
accumulate slightly different floating-point rounding — consistent with
the sub-0.0005 gaps observed in every differing case. This experiment did
not isolate that as the sole cause with a controlled test (e.g. feeding
identical float64 vectors through both paths), so it should be read as a
well-evidenced contributing factor, not a fully proven single explanation.
In every case observed, the affected candidates were tied or nearly tied;
top-1 and every aggregate metric were unaffected.

### Performance

CPU-only, single-machine, descriptive: model load 12.83s (shared by both
dense retrievers); local dense corpus embedding 206.99s; Qdrant collection
open 0.03s. Query retrieval time is about the same either way (dense local
6.57s / 50 queries, dense Qdrant 6.51s) — Qdrant only removes the corpus
embedding step, not per-query embedding, which still happens locally
through the same `SentenceTransformerEmbedder`. That corpus-embedding step
is exactly what a second `evaluate.py --retriever qdrant-dense` run
demonstrates disappearing: it completes in seconds, not ~200s, because it
only opens the existing collection and embeds 50 queries.

### Limitations specific to Phase 1D

- Qdrant native BM25 and native RRF (Query API fusion) were deliberately
  not used — this phase isolates dense storage/search as the only changed
  variable. Both are candidates for a future, separately controlled
  comparison against the current application-side BM25 + RRF.
- Local persistent mode does not represent a multi-process production
  deployment (concurrent readers/writers, network access); it validates
  storage and ranking parity, not deployment architecture.
- Parity was measured on the same 40-chunk, 50-query benchmark as every
  earlier phase — a larger corpus could behave differently, and this
  configuration's exact local-mode search doesn't test how parity holds up
  against Qdrant's approximate (HNSW) search, which is what a
  larger, server-mode deployment would actually use.

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

Phase 1D confirms Qdrant-backed storage preserves both the aggregate
metrics and the query-by-query ranking of Phase 1C's hybrid retriever (the
reason it's the retriever now persisted — see Phase 1C above for why it
beat dense), while eliminating the ~200s corpus re-embedding cost on every
process start. With dense storage no longer a confound, the next
controlled experiment is Qdrant's native BM25 and native RRF (Query API
fusion) against the current application-side BM25 + RRF, rather than
adding a KB-sufficiency threshold, reranking, or a second embedding model
before that comparison exists.
