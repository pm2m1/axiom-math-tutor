from axiom.retrieval.metrics import hit_at_k, recall_at_k, reciprocal_rank


def test_hit_at_k_true_when_relevant_in_window() -> None:
    ranked = ["c", "a", "b"]
    assert hit_at_k(ranked, {"a"}, k=3) is True
    assert hit_at_k(ranked, {"a"}, k=1) is False


def test_hit_at_k_false_when_nothing_relevant_retrieved() -> None:
    assert hit_at_k(["x", "y"], {"a"}, k=5) is False


def test_recall_at_k_counts_fraction_of_relevant_set_found() -> None:
    ranked = ["a", "x", "b", "y", "z"]
    assert recall_at_k(ranked, {"a", "b", "c"}, k=5) == 2 / 3


def test_recall_at_k_zero_when_no_relevant_ids_labeled() -> None:
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit() -> None:
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["a", "x"], {"a"}) == 1.0


def test_reciprocal_rank_zero_when_not_found() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
