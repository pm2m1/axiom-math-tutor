"""Cross-encoder evidence-support scoring (Phase 1H). Given a query and the
selected hybrid retriever's top-5 chunks — unchanged, no reranking, see
docs/kb-sufficiency.md — scores how relevant a query-passage cross-encoder
finds each one.

This is a relevance/ranking model, not an entailment model: a high score
means the passage is topically close to the query, not that the passage
actually contains enough information to answer it. Whether that relevance
signal is useful for the KB-sufficiency decision is exactly what this
phase measures, not something assumed going in.
"""

from dataclasses import dataclass
from typing import Protocol

CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Pinned so a future re-run scores against the exact weights this baseline
# was measured against, not whatever "main" resolves to later.
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class PairScorer(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """cross-encoder/ms-marco-MiniLM-L6-v2's own config declares
    `sbert_ce_default_activation_function: Identity` — `predict()` returns
    a raw, unbounded logit, not a probability. Higher means more relevant;
    there's no model-provided "this is definitely relevant" cutoff."""

    def __init__(
        self, model_id: str = CROSS_ENCODER_MODEL_ID, revision: str = CROSS_ENCODER_REVISION
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_id, revision=revision)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs: list[tuple[str, str]] = [(query, passage) for passage in passages]
        return [float(s) for s in self._model.predict(pairs)]


@dataclass
class EvidenceSupportFeatures:
    """ce_top1/ce_top2/ce_margin/ce_mean_top3 are computed over passages in
    the hybrid retriever's existing rank order — this scores fixed
    evidence, it does not rerank it. `vector()` returns only the two
    features used by the combined classifier (docs/kb-sufficiency.md);
    ce_top2 and ce_mean_top3 remain available for the separate scalar
    baselines."""

    ce_top1: float
    ce_top2: float
    ce_margin: float
    ce_mean_top3: float

    def vector(self) -> list[float]:
        return [self.ce_top1, self.ce_margin]


def extract_evidence_support(
    scorer: PairScorer, query: str, passages: list[str]
) -> EvidenceSupportFeatures:
    scores = scorer.score(query, passages)
    top1 = scores[0] if len(scores) >= 1 else 0.0
    top2 = scores[1] if len(scores) >= 2 else 0.0
    margin = top1 - top2 if len(scores) >= 2 else 0.0
    top3 = scores[:3]
    mean_top3 = sum(top3) / len(top3) if top3 else 0.0
    return EvidenceSupportFeatures(
        ce_top1=top1, ce_top2=top2, ce_margin=margin, ce_mean_top3=mean_top3
    )
