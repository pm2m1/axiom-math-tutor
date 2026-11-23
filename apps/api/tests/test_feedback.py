from pathlib import Path

import pytest

from axiom.feedback import FeedbackStore, UnknownAnswerError


def test_record_answer_returns_a_usable_id(tmp_path: Path) -> None:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    answer_id = store.record_answer("kb")
    assert store.get_answer(answer_id) is not None


def test_submit_feedback_persists_rating_and_correction(tmp_path: Path) -> None:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    answer_id = store.record_answer("kb")

    record = store.submit_feedback(answer_id, "up", "clear and correct")

    assert record.answer_id == answer_id
    assert record.rating == "up"
    assert record.correction == "clear and correct"
    assert record.answer_status == "kb"


def test_feedback_survives_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "fb.sqlite3"
    first = FeedbackStore(path=path)
    answer_id = first.record_answer("web")
    first.submit_feedback(answer_id, "down", "missed a case")

    reopened = FeedbackStore(path=path)
    record = reopened.get_feedback(answer_id)

    assert record is not None
    assert record.rating == "down"
    assert record.correction == "missed a case"


def test_resubmitting_feedback_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    answer_id = store.record_answer("kb")
    store.submit_feedback(answer_id, "up", "first take")

    store.submit_feedback(answer_id, "down", "changed my mind")

    with store._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE answer_id = ?", (answer_id,)
        ).fetchone()[0]
    assert count == 1
    record = store.get_feedback(answer_id)
    assert record is not None
    assert record.rating == "down"
    assert record.correction == "changed my mind"


def test_feedback_for_unknown_answer_id_is_rejected(tmp_path: Path) -> None:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    with pytest.raises(UnknownAnswerError):
        store.submit_feedback("not-a-real-answer-id", "up", None)


def test_get_feedback_returns_none_when_no_feedback_submitted(tmp_path: Path) -> None:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    answer_id = store.record_answer("kb")
    assert store.get_feedback(answer_id) is None


def test_different_answers_do_not_share_feedback(tmp_path: Path) -> None:
    store = FeedbackStore(path=tmp_path / "fb.sqlite3")
    first_id = store.record_answer("kb")
    second_id = store.record_answer("web")
    store.submit_feedback(first_id, "up", None)

    assert store.get_feedback(first_id) is not None
    assert store.get_feedback(second_id) is None
