from axiom.benchmark.harness import (
    DEV_PATH,
    FROZEN_PATH,
    BenchmarkItem,
    compute_metrics,
    evaluate_item,
    load_benchmark,
)


class TestLoadBenchmark:
    def test_loads_dev_set(self) -> None:
        items = load_benchmark(DEV_PATH)
        assert len(items) > 0
        assert all(item.source == "synthetic" for item in items)

    def test_loads_frozen_set(self) -> None:
        items = load_benchmark(FROZEN_PATH)
        assert len(items) > 0
        assert all(item.source == "synthetic" for item in items)

    def test_item_ids_are_unique_within_each_set(self) -> None:
        for path in (DEV_PATH, FROZEN_PATH):
            items = load_benchmark(path)
            ids = [item.id for item in items]
            assert len(ids) == len(set(ids))


class TestEvaluateItem:
    def test_equation_item_produces_two_checks(self) -> None:
        item = BenchmarkItem(
            id="t1",
            kind="equation",
            topic="linear_equations",
            source="synthetic",
            payload={
                "equation": "2*x = 8",
                "variable": "x",
                "correct_answer": ["4"],
                "incorrect_answer": ["5"],
            },
        )
        results = evaluate_item(item)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_unsupported_item_produces_one_check(self) -> None:
        item = BenchmarkItem(
            id="t2",
            kind="unsupported",
            topic="higher_degree_polynomial",
            source="synthetic",
            payload={"equation": "x^3 = 8", "variable": "x"},
        )
        results = evaluate_item(item)
        assert len(results) == 1
        assert results[0].passed

    def test_a_genuinely_wrong_dataset_label_is_caught(self) -> None:
        # Deliberately mislabel "5" as incorrect for 2*x=10 (it's actually
        # correct) — the harness must report this as a failed check, not
        # silently pass, since the whole point is catching authoring bugs.
        item = BenchmarkItem(
            id="t3",
            kind="equation",
            topic="linear_equations",
            source="synthetic",
            payload={
                "equation": "2*x = 10",
                "variable": "x",
                "correct_answer": ["5"],
                "incorrect_answer": ["5"],
            },
        )
        results = evaluate_item(item)
        incorrect_check = next(r for r in results if r.check_kind == "incorrect")
        assert incorrect_check.passed is False
        assert incorrect_check.actual_status == "verified_correct"


class TestComputeMetrics:
    def test_dev_set_is_100_percent_accurate_with_zero_false_verification(self) -> None:
        items = load_benchmark(DEV_PATH)
        results = [r for item in items for r in evaluate_item(item)]
        metrics = compute_metrics(results)

        assert metrics.accuracy == 1.0
        assert metrics.false_verification_rate == 0.0
        assert metrics.unsupported_correctly_identified_rate == 1.0

    def test_frozen_set_is_100_percent_accurate_with_zero_false_verification(self) -> None:
        items = load_benchmark(FROZEN_PATH)
        results = [r for item in items for r in evaluate_item(item)]
        metrics = compute_metrics(results)

        assert metrics.accuracy == 1.0
        assert metrics.false_verification_rate == 0.0
        assert metrics.unsupported_correctly_identified_rate == 1.0

    def test_latency_is_measured_and_small(self) -> None:
        items = load_benchmark(DEV_PATH)
        results = [r for item in items for r in evaluate_item(item)]
        metrics = compute_metrics(results)

        assert metrics.median_latency_seconds >= 0
        assert metrics.median_latency_seconds < 1.0

    def test_empty_results_do_not_divide_by_zero(self) -> None:
        metrics = compute_metrics([])
        assert metrics.accuracy == 0.0
        assert metrics.false_verification_rate == 0.0
