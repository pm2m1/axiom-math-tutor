from pathlib import Path

import pytest

from axiom.retrieval.evaluate import EvalCase, load_chunks, load_eval_cases


def test_real_corpus_and_eval_set_load_and_cross_reference() -> None:
    """Guards against the committed corpus and evaluation set drifting apart
    (e.g. a chunk_id referenced by a query no longer existing after a corpus
    regeneration)."""
    chunks = load_chunks()
    cases = load_eval_cases()
    chunk_ids = {c.chunk_id for c in chunks}

    assert chunks
    assert cases
    for case in cases:
        for relevant_id in case.relevant_chunk_ids:
            assert relevant_id in chunk_ids, f"{case.id} references unknown chunk {relevant_id}"


def test_out_of_scope_cases_have_no_relevant_chunks() -> None:
    cases = load_eval_cases()
    for case in cases:
        if not case.in_scope:
            assert case.relevant_chunk_ids == []


def test_eval_case_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x", "query": "q"}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_eval_cases(path)


def test_eval_case_rejects_wrong_type_for_in_scope() -> None:
    with pytest.raises(ValueError):
        EvalCase.model_validate_json(
            '{"id": "x", "query": "q", "relevant_chunk_ids": [], '
            '"topic": "t", "query_type": "definition", "in_scope": "not-a-boolean"}'
        )
