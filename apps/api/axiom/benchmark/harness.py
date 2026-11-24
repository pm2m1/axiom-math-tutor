"""Deterministic-solver benchmark harness (Phase 3).

    uv run python -m axiom.benchmark.harness --dataset dev
    uv run python -m axiom.benchmark.harness --dataset frozen

**What this measures, and what it does not.** This harness exercises
the exact SymPy solvers/verifiers in `axiom/verification/engine.py`
(equations, derivatives, indefinite integrals, combinatorics) against a
small, hand-checked item set — submitting each item's known-correct
answer (expecting `"verified_correct"`) and a known-incorrect one
(expecting `"verified_incorrect"`), plus a few deliberately
out-of-scope items (expecting `"unsupported"`). It is **not** a measure
of end-to-end JEE word-problem solving: there is no natural-language
pipeline here that reads a JEE question and extracts this structured
input automatically (see `docs/jee-solving.md` for why — briefly, that
would require either trusting an LLM's extraction as ground truth,
defeating the point of deterministic verification, or a much larger
NLP effort out of scope for this milestone). What is measured is
whether the *verifier itself* is reliable — for an exact symbolic
engine, accuracy should be 100% and the false-verification rate should
be exactly 0%; any deviation here is a genuine engine bug, not a
"model limitation" the way it would be for a fallible LLM judge.

**On the dataset.** `data/evaluation/jee_style_synthetic_{dev,frozen}.jsonl`
are hand-written by this project, in a JEE-adjacent style (linear/
quadratic equations, derivatives, integrals, combinatorics) — they are
explicitly **not** sourced from, and must never be described as, real
JEE Main/Advanced questions. A genuine, rights-cleared JEE dataset was
not established this phase; see `docs/jee-solving.md`'s "Benchmark
provenance" section for why, and do not report numbers from this
harness as JEE performance.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from axiom.verification.engine import (
    CombinatoricsOperation,
    check_combinatorics,
    check_derivative,
    check_integral,
    verify,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEV_PATH = _REPO_ROOT / "data" / "evaluation" / "jee_style_synthetic_dev.jsonl"
FROZEN_PATH = _REPO_ROOT / "data" / "evaluation" / "jee_style_synthetic_frozen.jsonl"

ItemKind = Literal["equation", "derivative", "integral", "combinatorics", "unsupported"]


@dataclass
class BenchmarkItem:
    id: str
    kind: ItemKind
    topic: str
    source: str
    payload: dict[str, object]


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = cast(dict[str, object], json.loads(line))
            items.append(
                BenchmarkItem(
                    id=str(raw["id"]),
                    kind=cast(ItemKind, raw["kind"]),
                    topic=str(raw["topic"]),
                    source=str(raw["source"]),
                    payload=raw,
                )
            )
    return items


CheckKind = Literal["correct", "incorrect", "unsupported"]


@dataclass
class CheckResult:
    item_id: str
    topic: str
    check_kind: CheckKind
    expected_status: str
    actual_status: str
    passed: bool
    latency_seconds: float


def _run_equation(item: BenchmarkItem, answer_key: str) -> tuple[str, float]:
    payload = item.payload
    answers = cast(list[str], payload[answer_key])
    t0 = time.perf_counter()
    outcome = verify(str(payload["equation"]), str(payload["variable"]), answers)
    return outcome.status, time.perf_counter() - t0


def _run_derivative(item: BenchmarkItem, answer_key: str) -> tuple[str, float]:
    payload = item.payload
    t0 = time.perf_counter()
    outcome = check_derivative(
        str(payload["expression"]), str(payload["variable"]), str(payload[answer_key])
    )
    return outcome.status, time.perf_counter() - t0


def _run_integral(item: BenchmarkItem, answer_key: str) -> tuple[str, float]:
    payload = item.payload
    t0 = time.perf_counter()
    outcome = check_integral(
        str(payload["expression"]), str(payload["variable"]), str(payload[answer_key])
    )
    return outcome.status, time.perf_counter() - t0


def _run_combinatorics(item: BenchmarkItem, answer_key: str) -> tuple[str, float]:
    payload = item.payload
    operation = cast(CombinatoricsOperation, payload["operation"])
    n = cast(int, payload["n"])
    r = cast(int, payload["r"])
    favorable = cast(int, payload["favorable"]) if "favorable" in payload else None
    t0 = time.perf_counter()
    outcome = check_combinatorics(operation, n, r, favorable, str(payload[answer_key]))
    return outcome.status, time.perf_counter() - t0


def _run_unsupported(item: BenchmarkItem) -> tuple[str, float]:
    payload = item.payload
    t0 = time.perf_counter()
    outcome = verify(str(payload["equation"]), str(payload["variable"]), [])
    return outcome.status, time.perf_counter() - t0


def evaluate_item(item: BenchmarkItem) -> list[CheckResult]:
    """One or two `CheckResult`s per item: `"unsupported"`-kind items
    get a single check (the engine must abstain); everything else gets
    both a correct-answer check (expect `"verified_correct"`) and an
    incorrect-answer check (expect `"verified_incorrect"`) — the
    adversarial pairing that would expose a false-verification bug."""
    if item.kind == "unsupported":
        status, latency = _run_unsupported(item)
        return [
            CheckResult(
                item.id,
                item.topic,
                "unsupported",
                "unsupported",
                status,
                status == "unsupported",
                latency,
            )
        ]

    runner = {
        "equation": _run_equation,
        "derivative": _run_derivative,
        "integral": _run_integral,
        "combinatorics": _run_combinatorics,
    }[item.kind]

    checks: list[tuple[CheckKind, str, str]] = [
        ("correct", "correct_answer", "verified_correct"),
        ("incorrect", "incorrect_answer", "verified_incorrect"),
    ]
    results = []
    for check_kind, answer_key, expected in checks:
        status, latency = runner(item, answer_key)
        results.append(
            CheckResult(
                item.id, item.topic, check_kind, expected, status, status == expected, latency
            )
        )
    return results


@dataclass
class BenchmarkMetrics:
    total_checks: int
    accuracy: float
    false_verification_rate: float
    unsupported_correctly_identified_rate: float
    by_topic_accuracy: dict[str, float]
    median_latency_seconds: float
    p95_latency_seconds: float


def compute_metrics(results: list[CheckResult]) -> BenchmarkMetrics:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total if total else 0.0

    incorrect_checks = [r for r in results if r.check_kind == "incorrect"]
    false_verifications = sum(1 for r in incorrect_checks if r.actual_status == "verified_correct")
    false_verification_rate = (
        false_verifications / len(incorrect_checks) if incorrect_checks else 0.0
    )

    unsupported_checks = [r for r in results if r.check_kind == "unsupported"]
    unsupported_correct = sum(1 for r in unsupported_checks if r.passed)
    unsupported_rate = unsupported_correct / len(unsupported_checks) if unsupported_checks else 0.0

    by_topic: dict[str, list[bool]] = {}
    for r in results:
        by_topic.setdefault(r.topic, []).append(r.passed)
    by_topic_accuracy = {topic: sum(v) / len(v) for topic, v in by_topic.items()}

    latencies = sorted(r.latency_seconds for r in results)
    median_latency = statistics.median(latencies) if latencies else 0.0
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    p95_latency = latencies[p95_index] if latencies else 0.0

    return BenchmarkMetrics(
        total_checks=total,
        accuracy=accuracy,
        false_verification_rate=false_verification_rate,
        unsupported_correctly_identified_rate=unsupported_rate,
        by_topic_accuracy=by_topic_accuracy,
        median_latency_seconds=median_latency,
        p95_latency_seconds=p95_latency,
    )


def run_benchmark(path: Path) -> BenchmarkMetrics:
    items = load_benchmark(path)
    results = [r for item in items for r in evaluate_item(item)]
    return compute_metrics(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["dev", "frozen"], required=True)
    args = parser.parse_args()
    path = DEV_PATH if args.dataset == "dev" else FROZEN_PATH
    metrics = run_benchmark(path)
    print(f"dataset: {args.dataset} ({path.name})")
    print(f"total checks: {metrics.total_checks}")
    print(f"accuracy: {metrics.accuracy:.1%}")
    print(f"false verification rate: {metrics.false_verification_rate:.1%}")
    print(f"unsupported correctly identified: {metrics.unsupported_correctly_identified_rate:.1%}")
    print(f"median latency: {metrics.median_latency_seconds * 1000:.2f} ms")
    print(f"p95 latency: {metrics.p95_latency_seconds * 1000:.2f} ms")
    print("by topic:")
    for topic, acc in sorted(metrics.by_topic_accuracy.items()):
        print(f"  {topic}: {acc:.1%}")


if __name__ == "__main__":
    main()
