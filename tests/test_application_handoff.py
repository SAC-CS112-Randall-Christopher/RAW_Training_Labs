"""Metadata/contract fixtures only; no model generation or optimizer steps."""

import json

import pytest
from conftest import reseal_export

from llm_lab import corpus
from llm_lab.corpus import import_qtrades, verify_corpus, write_corpus
from llm_lab.exposure import check, retire, safe_file
from llm_lab.fixtures import fixture_files
from llm_lab.io import canonical, read_object, sha_file


def policy(root, *, sealed=True, legacy=False):
    metadata = {
        "original": {
            "split": "test",
            "evidence_key": "same evidence",
            "families": ["original family"],
            "window": [100, 200],
            "data_basis": "observed",
        }
    }
    name = "metadata" if legacy else "exposure_metadata"
    path = root / "old-manifest.json"
    path.write_text(canonical({name: metadata}), encoding="utf-8")
    return {
        "studies": [
            {"id": "old study", "manifest": path.name, "sha256": sha_file(path), "sealed": sealed}
        ],
        "retirements": [],
    }


@pytest.mark.parametrize("changed", ["ids", "families", "window"])
def test_cross_study_protects_source_evidence_and_overlap_under_changed_ids(tmp_path, changed):
    p = policy(tmp_path)
    incoming = {
        "id": "renamed filename and new target",
        "split": "train",
        "evidence_key": "same evidence",
        "families": ["original family"],
        "window": [150, 160],
    }
    if changed == "families":
        incoming["families"] = ["renamed contrast"]
    elif changed == "window":
        incoming["evidence_key"] = "different BTC/ETH input"
        incoming["families"] = ["different instrument"]
    result = check(tmp_path, p, [incoming])
    assert not result["eligible"] and result["problems"][0]["code"] == "cross_study_exposure"


def test_legacy_families_remain_conservatively_protected_without_freshness_claim(tmp_path):
    p = policy(tmp_path, legacy=True)
    path = tmp_path / "old-manifest.json"
    value = read_object(path)
    del value["metadata"]["original"]["split"]
    path.write_text(canonical(value))
    p["studies"][0]["sha256"] = sha_file(path)
    result = check(
        tmp_path, p, [{"id": "new id", "families": ["original family"], "split": "train"}]
    )
    assert not result["eligible"] and result["limitations"]


def test_explicit_retirement_keeps_original_history_and_forbids_sealed_population(tmp_path):
    p = policy(tmp_path)
    request = {
        "study": "old study",
        "reviewer": "Fixture reviewer",
        "reason": "Retire this consumed fixture into regression material.",
        "splits": ["validation", "test", "unknown_legacy"],
    }
    original = (tmp_path / "old-manifest.json").read_bytes()
    with pytest.raises(ValueError, match="sealed"):
        retire(tmp_path, p, request, tmp_path / "retired.json")
    p["studies"][0]["sealed"] = False
    record = retire(tmp_path, p, request, tmp_path / "retired.json")
    p["retirements"] = ["retired.json"]
    result = check(tmp_path, p, [{"id": "renamed", "families": ["original family"]}])
    assert result["eligible"] and record["freshness"] == "retired_into_training_regression"
    assert (tmp_path / "old-manifest.json").read_bytes() == original
    fresh = check(
        tmp_path, p, [{"id": "renamed", "families": ["original family"], "split": "test"}]
    )
    assert not fresh["eligible"]


def test_old_training_cannot_become_fresh_evaluation_under_new_ids(tmp_path):
    p = policy(tmp_path, sealed=False)
    path = tmp_path / "old-manifest.json"
    value = read_object(path)
    value["exposure_metadata"]["original"]["split"] = "train"
    path.write_text(canonical(value))
    p["studies"][0]["sha256"] = sha_file(path)
    result = check(
        tmp_path, p, [{"id": "new ID", "families": ["original family"], "split": "validation"}]
    )
    assert not result["eligible"]


@pytest.mark.parametrize(
    "name", ["../outside.json", "http://example.invalid/data", "file:///secret"]
)
def test_unsafe_artifact_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_file(tmp_path, name)


def test_excluded_evaluation_targets_are_not_parsed_by_real_importer(
    source_export, tmp_path, monkeypatch
):
    source, contract = source_export
    manifest = read_object(source / "manifest.json")
    manifest["exposure_metadata"] = {"protected": {"split": "test", "families": ["sealed fixture"]}}
    (source / "manifest.json").write_text(canonical(manifest))
    reseal_export(source)
    read = corpus.read_rows
    calls = []

    def selected(path):
        calls.append(path.name)
        if path.name.startswith("test"):
            raise AssertionError("Protected final targets reached preparation")
        return read(path)

    monkeypatch.setattr(corpus, "read_rows", selected)
    imported = tmp_path / "imported"
    import_qtrades(source, imported, contract, intended_splits=("train", "validation"))
    _, files = verify_corpus(imported, splits=("train", "validation"))
    assert set(files) == {"train", "validation"}
    assert sha_file(imported / "test.jsonl") == sha_file(source / "test.jsonl")
    assert all(not name.startswith("test") for name in calls)


def test_selective_verification_keeps_test_and_regression_targets_out_of_worker(
    tmp_path, monkeypatch
):
    path = tmp_path / "corpus"
    files, metadata = fixture_files()
    write_corpus(files, path, {"source": "synthetic software fixture", "metadata": metadata})
    read = corpus.read_rows

    def selected(file):
        assert file.name in {"train.jsonl", "validation.jsonl"}
        return read(file)

    monkeypatch.setattr(corpus, "read_rows", selected)
    _, files = verify_corpus(path, splits=("train", "validation"))
    assert set(files) == {"train", "validation"}


def test_legacy_approval_is_preserved_but_cannot_be_newly_imported_as_human(
    source_export, tmp_path
):
    source, contract = source_export
    path = source / "train.provenance.jsonl"
    original = json.loads(path.read_text())
    del original["review"]["reviewer_kind"]
    path.write_text(canonical(original) + "\n")
    reseal_export(source)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Legacy reviewer"):
        import_qtrades(source, tmp_path / "new", contract)
    assert path.read_bytes() == before


def test_review_link_must_match_imported_original_not_another_saved_target(source_export, tmp_path):
    source, contract = source_export
    manifest = read_object(source / "manifest.json")
    manifest["source_reviews"] = {}
    for split in ("train", "validation", "test"):
        origin = corpus.read_rows(source / (split + ".provenance.jsonl"))[0]
        manifest["source_reviews"][origin["candidate"]["candidate_sha256"]] = {
            "revision": 1,
            "review_sha256": "different review",
            "target_sha256": "different target",
        }
    (source / "manifest.json").write_text(canonical(manifest))
    reseal_export(source)
    with pytest.raises(ValueError, match="review revision"):
        import_qtrades(source, tmp_path / "new", contract)
    assert not (tmp_path / "new").exists()


def test_real_training_preprocessing_excludes_final_targets_before_any_model_load(
    tmp_path, monkeypatch
):
    from llm_lab import training
    from llm_lab.config import Recipe

    files, metadata = fixture_files()
    directory = tmp_path / "corpus"
    write_corpus(files, directory, {"source": "synthetic fixture", "metadata": metadata})
    calls = []
    read = corpus.read_rows

    def selected(file):
        assert file.name in {"train.jsonl", "validation.jsonl"}
        calls.append(file.name)
        return read(file)

    monkeypatch.setattr(corpus, "read_rows", selected)
    monkeypatch.setattr(training, "verify_model", lambda base: {"base_sha256": "fixture"})
    monkeypatch.setattr(training, "tokenizer_at", lambda base: object())
    monkeypatch.setattr(training, "low_priority", lambda: None)

    def boundary(values, tokenizer, length, **kwargs):
        assert set(values) == {"train", "validation"}
        raise ValueError("Fixture stops at preprocessing; no model or optimizer")

    monkeypatch.setattr(training, "preflight", boundary)
    with pytest.raises(ValueError, match="stops at preprocessing"):
        training.train(
            tmp_path / "unused-base",
            directory,
            Recipe(
                name="preprocessing-only-fixture",
                device="cpu",
                precision="float32",
                quantization="none",
            ),
            tmp_path / "no-run",
        )
    assert calls == ["train.jsonl", "validation.jsonl"]
    assert not (tmp_path / "no-run").exists()


@pytest.mark.parametrize("wrong", ["dataset", "model", "profile", "run", "review_source"])
def test_preserved_comparison_rejects_conflicting_original_linkage(source_export, tmp_path, wrong):
    from llm_lab.comparison_receipt import reopen
    from llm_lab.io import write_new

    source, contract = source_export
    manifest = import_qtrades(source, tmp_path / "corpus", contract)
    run = {
        "status": "trained",
        "base_sha256": "base",
        "profile_sha256": "profile",
        "corpus_sha256": manifest["corpus_sha256"],
        "corpus_directory": str(tmp_path / "corpus"),
    }
    plan = {
        "candidate_profile_sha256": "profile",
        "training_corpus_sha256": run["corpus_sha256"],
        "corpus_sha256": "evaluation",
    }
    mechanical = {"candidate_profile_sha256": "profile", "corpus_sha256": "evaluation"}
    (tmp_path / "answers.jsonl").write_text("Synthetic frozen answers fixture; never parsed\n")
    for name, value in [("run", run), ("plan", plan), ("comparison", mechanical)]:
        write_new(tmp_path / (name + ".json"), value)
    semantic = {
        "source": {
            "comparison_sha256": sha_file(tmp_path / "comparison.json"),
            "responses_sha256": sha_file(tmp_path / "answers.jsonl"),
        }
    }
    if wrong == "review_source":
        semantic["source"]["responses_sha256"] = "other answers"
    write_new(tmp_path / "semantic.json", semantic)
    link = {
        "id": "fixture",
        "model_sha256": "other" if wrong == "model" else "base",
        "dataset_sha256": "other" if wrong == "dataset" else manifest["source_corpus_sha256"],
    }
    for name in ("run", "plan", "comparison", "semantic", "answers"):
        path = tmp_path / (name + (".jsonl" if name == "answers" else ".json"))
        link[name], link[name + "_sha256"] = path.name, sha_file(path)
    if wrong == "run":
        link["run_sha256"] = "changed source file"
    if wrong == "profile":
        plan["candidate_profile_sha256"] = "other profile"
        (tmp_path / "plan.json").write_text(canonical(plan))
        link["plan_sha256"] = sha_file(tmp_path / "plan.json")
    with pytest.raises(ValueError, match="changed|expected|different|differs"):
        reopen(tmp_path, link)
