from pathlib import Path

from axiom.mastery import DEFAULT_MASTERY_ESTIMATE, LEARNING_RATE, MasteryStore


def test_unattempted_skill_returns_the_default_estimate(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    assert store.get_mastery("learner-1", "linear_equations") == DEFAULT_MASTERY_ESTIMATE


def test_correct_attempt_increases_the_estimate(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    result = store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    expected = DEFAULT_MASTERY_ESTIMATE + LEARNING_RATE * (1.0 - DEFAULT_MASTERY_ESTIMATE)
    assert result.estimate == expected
    assert result.estimate > DEFAULT_MASTERY_ESTIMATE


def test_incorrect_attempt_decreases_the_estimate(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    result = store.record_attempt("learner-1", "linear_equations", verified=True, correct=False)
    assert result.estimate < DEFAULT_MASTERY_ESTIMATE


def test_estimate_is_clamped_between_zero_and_one(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    for _ in range(50):
        store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    result = store.get_mastery("learner-1", "linear_equations")
    assert 0.0 <= result <= 1.0


def test_unverified_attempt_does_not_change_the_estimate(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    after_verified = store.get_mastery("learner-1", "linear_equations")

    store.record_attempt("learner-1", "linear_equations", verified=False, correct=None)
    after_unverified = store.get_mastery("learner-1", "linear_equations")

    assert after_unverified == after_verified


def test_verified_but_indeterminate_outcome_does_not_change_the_estimate(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    after_first = store.get_mastery("learner-1", "linear_equations")

    # "verified" pipeline ran, but correctness itself was indeterminate —
    # correct=None must still withhold the update.
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=None)
    after_second = store.get_mastery("learner-1", "linear_equations")

    assert after_second == after_first


def test_mastery_is_tracked_independently_per_skill(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    assert store.get_mastery("learner-1", "quadratic_equations") == DEFAULT_MASTERY_ESTIMATE


def test_mastery_is_tracked_independently_per_learner(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    assert store.get_mastery("learner-2", "linear_equations") == DEFAULT_MASTERY_ESTIMATE


def test_get_all_mastery_lists_every_attempted_skill(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    store.record_attempt("learner-1", "quadratic_equations", verified=True, correct=False)

    skills = {m.skill for m in store.get_all_mastery("learner-1")}
    assert skills == {"linear_equations", "quadratic_equations"}


def test_mastery_survives_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"
    store = MasteryStore(path)
    store.record_attempt("learner-1", "linear_equations", verified=True, correct=True)
    expected = store.get_mastery("learner-1", "linear_equations")

    reopened = MasteryStore(path)
    assert reopened.get_mastery("learner-1", "linear_equations") == expected


def test_hints_used_is_logged_but_does_not_affect_the_estimate(tmp_path: Path) -> None:
    store = MasteryStore(tmp_path / "m.sqlite3")
    with_hints = store.record_attempt(
        "learner-1", "linear_equations", verified=True, correct=True, hints_used=3
    )
    store2 = MasteryStore(tmp_path / "m2.sqlite3")
    without_hints = store2.record_attempt(
        "learner-1", "linear_equations", verified=True, correct=True, hints_used=0
    )
    assert with_hints.estimate == without_hints.estimate
