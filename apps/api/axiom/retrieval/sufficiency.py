"""KB-sufficiency routing (Phase 1G): given retrieval evidence for a query,
decide whether the current algebra corpus actually supports answering it,
or whether the question should fall back to something outside the KB (web
search / MCP, not implemented yet — this module only produces the routing
decision). See docs/kb-sufficiency.md.

Uses the selected retrieval pipeline unchanged: rank_bm25.BM25Retriever +
Qdrant dense retrieval, plus the existing application-side RRF for a fused
score. Nothing here retunes retrieval — chunking, tokenization, embedding
model, and RRF k/depth are exactly as validated through Phase 1F.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from axiom.retrieval.base import Retriever

_REPO_ROOT = Path(__file__).resolve().parents[4]
SUFFICIENCY_PATH = _REPO_ROOT / "data" / "evaluation" / "algebra_sufficiency.jsonl"
# Phase 1H fresh holdout — created only after the Phase 1H model/feature/
# threshold design was frozen against the (already-inspected) set above.
SUFFICIENCY_V2_PATH = _REPO_ROOT / "data" / "evaluation" / "algebra_sufficiency_v2.jsonl"


class SufficiencyCase(BaseModel):
    id: str
    query: str
    supported: bool
    support_chunk_ids: list[str]
    negative_type: str | None
    topic: str
    split: str


def load_sufficiency_cases(path: Path = SUFFICIENCY_PATH) -> list[SufficiencyCase]:
    with path.open(encoding="utf-8") as f:
        return [SufficiencyCase.model_validate_json(line) for line in f if line.strip()]


@dataclass
class RetrievalEvidence:
    bm25_hits_ids: list[str]
    bm25_scores: list[float]
    dense_hits_ids: list[str]
    dense_scores: list[float]
    hybrid_hits_ids: list[str]
    hybrid_scores: list[float]
    hybrid_texts: list[str]


def gather_evidence(
    bm25: Retriever, dense: Retriever, hybrid: Retriever, query: str, *, depth: int = 5
) -> RetrievalEvidence:
    bm25_hits = bm25.search(query, limit=depth)
    dense_hits = dense.search(query, limit=depth)
    hybrid_hits = hybrid.search(query, limit=depth)
    return RetrievalEvidence(
        bm25_hits_ids=[h.chunk.chunk_id for h in bm25_hits],
        bm25_scores=[h.score for h in bm25_hits],
        dense_hits_ids=[h.chunk.chunk_id for h in dense_hits],
        dense_scores=[h.score for h in dense_hits],
        hybrid_hits_ids=[h.chunk.chunk_id for h in hybrid_hits],
        hybrid_scores=[h.score for h in hybrid_hits],
        hybrid_texts=[h.chunk.text for h in hybrid_hits],
    )


@dataclass
class SufficiencyFeatures:
    """Six features from BM25 + dense retrieval evidence, matching the
    small set docs/kb-sufficiency.md documents a reason for. rrf_top1 is
    kept separate (used only by the RRF scalar baseline, not the
    multi-feature classifier) since it was previously found to have poor
    KB-sufficiency separation (Phase 1B/1C/1D confidence investigations)."""

    bm25_top1: float
    bm25_margin: float
    dense_top1: float
    dense_margin: float
    top1_agreement: float
    top5_overlap: float
    rrf_top1: float

    def vector(self) -> list[float]:
        return [
            self.bm25_top1,
            self.bm25_margin,
            self.dense_top1,
            self.dense_margin,
            self.top1_agreement,
            self.top5_overlap,
        ]


FEATURE_NAMES = [
    "bm25_top1",
    "bm25_margin",
    "dense_top1",
    "dense_margin",
    "top1_agreement",
    "top5_overlap",
]


def _top1(scores: list[float]) -> float:
    return scores[0] if scores else 0.0


def _margin(scores: list[float]) -> float:
    return scores[0] - scores[1] if len(scores) >= 2 else 0.0


def extract_features(evidence: RetrievalEvidence) -> SufficiencyFeatures:
    bm25_ids = set(evidence.bm25_hits_ids)
    dense_ids = set(evidence.dense_hits_ids)
    union = bm25_ids | dense_ids
    overlap = len(bm25_ids & dense_ids) / len(union) if union else 0.0
    top1_agree = (
        1.0
        if evidence.bm25_hits_ids
        and evidence.dense_hits_ids
        and evidence.bm25_hits_ids[0] == evidence.dense_hits_ids[0]
        else 0.0
    )
    return SufficiencyFeatures(
        bm25_top1=_top1(evidence.bm25_scores),
        bm25_margin=_margin(evidence.bm25_scores),
        dense_top1=_top1(evidence.dense_scores),
        dense_margin=_margin(evidence.dense_scores),
        top1_agreement=top1_agree,
        top5_overlap=overlap,
        rrf_top1=_top1(evidence.hybrid_scores),
    )


class SufficiencyDecision(BaseModel):
    """The routing decision this phase evaluates. `score` is a
    decision_score, not a calibrated probability — do not read it as
    certainty. No web/MCP fallback happens yet; this is the evaluation
    layer only."""

    use_knowledge_base: bool
    score: float
    strategy: str


@dataclass
class ConfusionMatrix:
    """KB-acceptance framed as the positive class. false_accept_rate is
    computed over actually-unsupported cases (how often an insufficient
    corpus still got USE_KB — the dangerous error); false_reject_rate is
    computed over actually-supported cases (how often sufficient evidence
    still got rejected — the safe-but-costly error)."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_accept_rate(self) -> float:
        denom = self.false_positive + self.true_negative
        return self.false_positive / denom if denom else 0.0

    @property
    def false_reject_rate(self) -> float:
        denom = self.false_negative + self.true_positive
        return self.false_negative / denom if denom else 0.0


def confusion_matrix(labels: list[bool], predictions: list[bool]) -> ConfusionMatrix:
    tp = sum(1 for label, pred in zip(labels, predictions, strict=True) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predictions, strict=True) if not label and pred)
    tn = sum(1 for label, pred in zip(labels, predictions, strict=True) if not label and not pred)
    fn = sum(1 for label, pred in zip(labels, predictions, strict=True) if label and not pred)
    return ConfusionMatrix(tp, fp, tn, fn)


@dataclass
class ThresholdSelection:
    threshold: float
    met_precision_target: bool
    dev_precision: float
    dev_recall: float


def select_threshold(
    labels: list[bool], scores: list[float], *, min_precision: float = 0.9
) -> ThresholdSelection:
    """Sweep every development score as a candidate accept-if-score>=t
    threshold. Among thresholds reaching KB-accept precision >=
    min_precision, keep the one with the highest supported-query recall.
    If none clears it, fall back to the threshold with the single highest
    precision observed, explicitly marked as not meeting the target —
    reported honestly rather than forcing an arbitrary cutoff."""
    candidates = sorted(set(scores), reverse=True)
    best_meeting_target: tuple[float, float, float] | None = None  # (threshold, precision, recall)
    best_overall: tuple[float, float, float] | None = None

    for threshold in candidates:
        predictions = [score >= threshold for score in scores]
        matrix = confusion_matrix(labels, predictions)
        candidate = (threshold, matrix.precision, matrix.recall)
        if best_overall is None or candidate[1] > best_overall[1]:
            best_overall = candidate
        if matrix.precision >= min_precision:
            if best_meeting_target is None or candidate[2] > best_meeting_target[2]:
                best_meeting_target = candidate

    if best_meeting_target is not None:
        threshold, precision, recall = best_meeting_target
        return ThresholdSelection(threshold, True, precision, recall)

    assert best_overall is not None  # candidates is non-empty for any real dataset
    threshold, precision, recall = best_overall
    return ThresholdSelection(threshold, False, precision, recall)


class LogisticSufficiencyModel:
    """Small interpretable classifier over plain feature vectors — no query
    text, only retrieval/evidence-support evidence. Takes vectors rather
    than a fixed feature type so the same wrapper serves both the Phase 1G
    retrieval-only model (SufficiencyFeatures.vector()) and Phase 1H's
    combined retrieval + cross-encoder model. Trained on development data
    only; features are standardized since they're on different scales
    (BM25 scores are unbounded, cosine/agreement/overlap/CE logits aren't
    all on the same scale either)."""

    def __init__(self, *, random_state: int = 0) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(random_state=random_state)

    def fit(self, feature_vectors: list[list[float]], labels: list[bool]) -> None:
        x = self._scaler.fit_transform(feature_vectors)
        self._model.fit(x, labels)

    def decision_scores(self, feature_vectors: list[list[float]]) -> list[float]:
        x = self._scaler.transform(feature_vectors)
        probabilities: Any = self._model.predict_proba(x)
        return [float(p[1]) for p in probabilities]


def decide(score: float, threshold: float, strategy: str) -> SufficiencyDecision:
    return SufficiencyDecision(
        use_knowledge_base=score >= threshold, score=score, strategy=strategy
    )
