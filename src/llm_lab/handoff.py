"""Existing importer and local tokenizer preparation, without a model worker."""

import time
from pathlib import Path

from llm_lab.config import Recipe
from llm_lab.corpus import import_qtrades, verify_corpus
from llm_lab.exposure import check, safe_file
from llm_lab.io import digest, read_object, sha_file, write_new
from llm_lab.masking import preflight
from llm_lab.model import source_identity, tokenizer_at, verify_model


def execute(root: Path, job: Path) -> dict:
    if not job.resolve().is_relative_to(root.resolve()) or job.is_symlink():
        raise ValueError("Handoff job is outside private storage")
    request = read_object(job / "request.json")
    if request.get("format") != "qtrades-lab-handoff-v1":
        raise ValueError("Unsupported handoff request")
    if request["operation"] == "exposure":
        result = check(root, request["policy"], request["metadata"])
    elif request["operation"] == "comparison":
        from llm_lab.comparison_receipt import reopen

        result = reopen(root, request["comparison_link"])
    elif request["operation"] == "retire":
        from llm_lab.exposure import retire

        result = retire(root, request["policy"], request["retirement"], job / "retirement.json")
    elif request["operation"] == "prepare":
        started = time.monotonic()
        expected = request["expected"]
        if request["profile"]["name"] != "Qwen3.5-4B":
            raise ValueError("This application handoff permits the selected 4B profile only")
        base = safe_file(root, request["profile"]["base"])
        recipe_path = safe_file(root, request["profile"]["recipe"])
        if sha_file(recipe_path) != request["profile"]["recipe_file_sha256"]:
            raise ValueError("Configured preparation recipe changed")
        model = verify_model(base)
        if model["base_sha256"] != expected["model_sha256"]:
            raise ValueError("Pinned base identity differs from the expected model")
        recipe = Recipe.model_validate(read_object(recipe_path))
        imported = import_qtrades(
            job / "bundle",
            job / "corpus",
            expected["contract_sha256"],
            intended_splits=("train", "validation"),
        )
        if imported["source_corpus_sha256"] != expected["dataset_sha256"]:
            raise ValueError("Imported source differs from the expected dataset")
        if imported["source_reviews"] != expected.get("source_reviews", {}):
            raise ValueError("Imported reviews differ from the expected saved revisions")
        if set(imported["exposure_metadata"]) != set(expected["source_examples"]):
            raise ValueError("Imported examples differ from the expected source membership")
        exposure = check(root, request["policy"], list(imported["exposure_metadata"].values()))
        if not exposure["eligible"]:
            raise ValueError("Cross-study protected exposure; no preparation accepted")
        manifest, files = verify_corpus(job / "corpus", splits=("train", "validation"))
        selected = {s: rows for s, rows in files.items() if rows}
        mask, _ = preflight(
            selected, tokenizer_at(base), recipe.max_length, enable_thinking=recipe.enable_thinking
        )
        result = {
            "format": "llm-lab-preparation-receipt-v1",
            "stage": "prepared",
            "study_id": expected["study_id"],
            "run_id": expected["run_id"],
            "dataset_sha256": expected["dataset_sha256"],
            "corpus_sha256": manifest["corpus_sha256"],
            "model_sha256": expected["model_sha256"],
            "profile_sha256": digest(request["profile"]),
            "contract_sha256": expected["contract_sha256"],
            "source_examples": expected["source_examples"],
            "source_reviews": imported["source_reviews"],
            "nonce": request["nonce"],
            "request_sha256": digest(request),
            "lab_source_sha256": source_identity()["source_sha256"],
            "masking": mask,
            "elapsed_seconds": time.monotonic() - started,
            "exposure": exposure,
            "trained": False,
            "evaluated": False,
            "limitations": [
                "Prepared using the actual pinned tokenizer; no weights loaded or trained.",
                "Only intended training/validation targets parsed; test targets excluded.",
                "No qualification, activation, causal contribution or trading improvement.",
            ]
            + exposure["limitations"],
        }
    else:
        raise ValueError("Unsupported bounded handoff operation")
    write_new(job / "receipt.json", result)
    return result
