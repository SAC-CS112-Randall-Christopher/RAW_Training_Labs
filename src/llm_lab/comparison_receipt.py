"""Reopen existing paired/adjudicated results with their original identity bindings."""

from pathlib import Path

from llm_lab.corpus import verify_corpus
from llm_lab.exposure import safe_file
from llm_lab.io import read_object, sha_file


def reopen(root: Path, link: dict) -> dict:
    paths = {
        k: safe_file(root, link[k]) for k in ("run", "plan", "comparison", "semantic", "answers")
    }
    for key, path in paths.items():
        if sha_file(path) != link[key + "_sha256"]:
            raise ValueError("Original comparison/review archive changed")
    run, plan, mechanical, semantic = (
        read_object(paths[k]) for k in ("run", "plan", "comparison", "semantic")
    )
    if (
        run["status"] != "trained"
        or run["base_sha256"] != link["model_sha256"]
        or plan["candidate_profile_sha256"] != run["profile_sha256"]
        or plan["training_corpus_sha256"] != run["corpus_sha256"]
        or mechanical["candidate_profile_sha256"] != run["profile_sha256"]
        or mechanical["corpus_sha256"] != plan["corpus_sha256"]
        or semantic["source"]["comparison_sha256"] != link["comparison_sha256"]
        or semantic["source"]["responses_sha256"] != link["answers_sha256"]
    ):
        raise ValueError("Comparison does not belong to the expected trained dataset/model/run")
    corpus_path = Path(run["corpus_directory"])
    if not corpus_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Run corpus is outside approved private provenance")
    manifest, _ = verify_corpus(corpus_path, splits=())
    if manifest["corpus_sha256"] != run["corpus_sha256"]:
        raise ValueError("Frozen training corpus changed")
    source_dataset = manifest.get("source_corpus_sha256", run["corpus_sha256"])
    if source_dataset != link["dataset_sha256"]:
        raise ValueError("Comparison dataset differs from its configured source")
    history = []
    for record in link.get("review_history", []):
        path = safe_file(root, record["path"])
        if sha_file(path) != record["sha256"]:
            raise ValueError("Original review/correction history changed")
        value = read_object(path)
        if "summaries" in value:
            if (
                value.get("source", {}).get("comparison_sha256") != link["comparison_sha256"]
                or value["source"].get("responses_sha256") != link["answers_sha256"]
            ):
                raise ValueError("Review history belongs to different original responses")
            value = {k: value[k] for k in ("summaries", "paired_changes", "limitations")}
        history.append({"label": record["label"], "sha256": record["sha256"], "record": value})
    return {
        "format": "llm-lab-linked-comparison-v1",
        "stage": "evaluated",
        "model_name": "Qwen3.5-4B / preserved v2",
        "model_sha256": run["base_sha256"],
        "profile_sha256": run["profile_sha256"],
        "dataset_sha256": source_dataset,
        "corpus_sha256": run["corpus_sha256"],
        "run_id": link["id"],
        "source_examples": list(manifest.get("metadata", {})),
        "trained": True,
        "evaluated": True,
        "comparison": {
            "evidence_label": "Reused historical instructional comparison; independent model "
            "review, not human adjudication",
            "semantic": semantic["summaries"],
            "mechanical": mechanical["summary"],
            "review_provenance": semantic["assessment"],
            "paired_changes": semantic["paired_changes"],
            "admitted": "Unavailable in this historical record",
            "judgments_sha256": semantic["judgments_sha256"],
            "archive_hashes": {k: link[k + "_sha256"] for k in paths},
            "case_judgments": [
                {
                    k: row[k]
                    for k in (
                        "arm",
                        "blind_id",
                        "case_id",
                        "components",
                        "failure_categories",
                        "family_ids",
                        "role",
                        "usable",
                    )
                }
                for row in semantic["cases"]
            ],
            "capability_and_comparison_dimensions": "Not separately adjudicated in this "
            "historical rubric; unavailable",
            "flags_disagreements_corrections": semantic.get(
                "reconciliation",
                "Original flags and corrections remain in linked private source history; "
                "no new adjudication",
            ),
            "preserved_review_history": history,
        },
        "limitations": semantic["limitations"]
        + [
            "Belongs to its original training corpus, never the new preparation dataset.",
            "Missing calls are coverage gaps; families and repeated review are "
            "not independent samples.",
            "Usability requires semantic judgment; native JSON and action matches are mechanical.",
            "No qualification, activation, individual-example causality or market improvement.",
        ],
    }
