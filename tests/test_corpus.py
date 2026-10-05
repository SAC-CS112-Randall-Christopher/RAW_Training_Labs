import copy

import pytest
from conftest import reseal_export

from llm_lab.corpus import import_qtrades, verify_corpus, write_corpus
from llm_lab.fixtures import fixture_files
from llm_lab.io import canonical, read_object, read_rows, write_state


def test_delegated_instructional_review_cannot_admit_an_empirical_episode(source_export, tmp_path):
    source, contract = source_export
    path = source / "train.provenance.jsonl"
    origin = read_rows(path)[0]
    origin["review"]["reviewer_kind"] = "delegated_semantic"
    path.write_text(canonical(origin) + "\n", encoding="utf-8")
    reseal_export(source)
    with pytest.raises(ValueError, match="empirical"):
        import_qtrades(source, tmp_path / "imported", contract)


def test_verified_source_import_retains_original_identity_and_missing_regression(
    source_export, tmp_path
):
    source, contract = source_export
    original = read_object(source / "manifest.json")
    dest = tmp_path / "imported"
    manifest = import_qtrades(source, dest, contract)
    _, files = verify_corpus(dest)
    assert files["train"] == read_rows(source / "train.jsonl")
    assert manifest["source_corpus_sha256"] == original["corpus_sha256"]
    assert manifest["regression_available"] is False
    assert "regression" not in files


def test_wrong_application_contract_refused(source_export, tmp_path):
    with pytest.raises(ValueError, match="contract"):
        import_qtrades(source_export[0], tmp_path / "imported", "wrong")


@pytest.mark.parametrize(
    "field, value",
    [
        ("rights_confirmed", False),
        ("approved", False),
        ("target_available_at", 101.0),
        ("reviewed_at", 99.0),
        ("episode_end", 105.0),
        ("evidence_available_at", {"e0": 101.0}),
        ("family_ids", []),
    ],
)
def test_resealed_future_or_unreviewed_source_still_refused(source_export, tmp_path, field, value):
    source, contract = source_export
    path = source / "train.provenance.jsonl"
    origin = read_rows(path)[0]
    origin["review"][field] = value
    path.write_text(canonical(origin) + "\n", encoding="utf-8")
    reseal_export(source)
    with pytest.raises(ValueError):
        import_qtrades(source, tmp_path / "imported", contract)


def test_resealed_family_crossing_split_refused(source_export, tmp_path):
    source, contract = source_export
    path = source / "validation.provenance.jsonl"
    origin = read_rows(path)[0]
    origin["review"]["family_ids"] = ["train"]
    path.write_text(canonical(origin) + "\n", encoding="utf-8")
    reseal_export(source)
    with pytest.raises(ValueError, match="family"):
        import_qtrades(source, tmp_path / "imported", contract)


def test_changed_target_refused_even_if_outer_hashes_resealed(source_export, tmp_path):
    source, contract = source_export
    path = source / "train.jsonl"
    row = read_rows(path)[0]
    row["completion"][0]["content"] = '{"action":"approve"}'
    path.write_text(canonical(row) + "\n", encoding="utf-8")
    reseal_export(source)
    with pytest.raises(ValueError, match="reviewed target"):
        import_qtrades(source, tmp_path / "imported", contract)


def test_changed_file_refused_before_any_output(source_export, tmp_path):
    source, contract = source_export
    with (source / "test.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Changed exported"):
        import_qtrades(source, tmp_path / "imported", contract)
    assert not (tmp_path / "imported").exists()


def test_corpus_never_reuses_test_prompt_in_training(tmp_path):
    files, meta = fixture_files()
    files["test"][0]["prompt"] = copy.deepcopy(files["train"][0]["prompt"])
    write_corpus(files, tmp_path / "corpus", {"metadata": meta})
    with pytest.raises(ValueError, match="prompt crosses"):
        verify_corpus(tmp_path / "corpus")


def test_changed_frozen_corpus_manifest_refused(hybrid, tmp_path):
    files, meta = fixture_files()
    write_corpus(files, tmp_path / "corpus", {"metadata": meta})
    manifest = read_object(tmp_path / "corpus" / "manifest.json")
    manifest["metadata"] = {}
    write_state(tmp_path / "corpus" / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        verify_corpus(tmp_path / "corpus")
