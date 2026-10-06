"""Frozen corpus ingestion. Applications own semantic adjudication and qualification."""

from collections import Counter
from pathlib import Path

from llm_lab.io import (
    canonical,
    digest,
    new_directory,
    qtrades_digest,
    read_object,
    read_rows,
    sha_file,
    write_new,
)

SPLITS = ("train", "validation", "test", "regression")


def check_row(row: dict) -> None:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Each example needs an original identity")
    if not isinstance(row.get("prompt"), list) or not row["prompt"]:
        raise ValueError("Each example needs its original conversation")
    if not isinstance(row.get("completion"), list) or len(row["completion"]) != 1:
        raise ValueError("Use one explicitly reviewed final assistant response")
    for message in row["prompt"] + row["completion"]:
        if not isinstance(message, dict):
            raise ValueError("Messages must be objects")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported conversation role")
        if not isinstance(message.get("content"), str):
            raise ValueError("Text-only messages must have string content")
    if row["completion"][0]["role"] != "assistant" or not row["completion"][0]["content"].strip():
        raise ValueError("The reviewed final response must be nonempty")
    if row["prompt"][-1]["role"] == "assistant":
        raise ValueError("Final response must follow an input or a tool result")


def _review_qtrades(row: dict, provenance: dict, contract: str) -> None:
    c, r = provenance["candidate"], provenance["review"]
    if c["contract_sha256"] != contract or c["candidate_sha256"] != row["id"]:
        raise ValueError("Original candidate or role contract changed")
    body = {k: v for k, v in c.items() if k != "candidate_sha256"}
    if qtrades_digest(body) != c["candidate_sha256"]:
        raise ValueError("Original candidate integrity failed")
    if qtrades_digest(c["packet"]) != c["packet_sha256"]:
        raise ValueError("Original packet integrity failed")
    if r["approved"] is not True or r["rights_confirmed"] is not True:
        raise ValueError("Review and data rights must be explicitly approved")
    instructional = c.get("source_kind", "model_attempt") == "instructional"
    if instructional != (r["data_basis"] == "instructional"):
        raise ValueError("Instructional provenance cannot substitute for empirical evidence")
    if instructional:
        authorship = c.get("authorship") or {}
        if (
            authorship.get("empirical_performance_claim") is not False
            or c["attempt"] != 0
            or c["original_answer"] is not None
            or not c["task_id"].startswith("authored:")
            or authorship.get("authored_at") != c["decision_at"]
            or c["finished_at"] != c["decision_at"]
            or not authorship.get("rights_basis")
        ):
            raise ValueError(
                "Instructional authorship is incomplete or imitates an operating episode"
            )
    observed = c.get("source_kind") == "observed_episode"
    if "reviewer_kind" not in r:
        raise ValueError("Legacy reviewer provenance is unresolved; no human default")
    if r["reviewer_kind"] not in {"human", "delegated_semantic"}:
        raise ValueError("Unknown reviewer type")
    if not isinstance(r.get("reviewer_authored_material"), bool) or not r.get("claim_scope"):
        raise ValueError("New approvals require reviewer authorship and claim scope")
    if observed:
        original = c.get("observed_source") or {}
        basis = "synthetic" if original.get("evidence_basis") == "synthetic_qa" else "observed"
        if (
            c["original_answer"] is not None
            or c["attempt"] != 0
            or r["claim_scope"] != "interpretation"
            or r["data_basis"] != basis
            or original.get("available_at", float("inf")) > c["decision_at"]
            or (r["episode_start"], r["episode_end"])
            != (original.get("observed_start"), original.get("observed_end"))
            or any(t < original["available_at"] for t in r["evidence_available_at"].values())
        ):
            raise ValueError("Observed episode supports original-timing interpretation only")
    if r["reviewer_kind"] == "delegated_semantic" and not (
        instructional or (observed and r["claim_scope"] == "interpretation")
    ):
        raise ValueError("Delegated instructional review cannot admit empirical training examples")
    if not r["family_ids"] or not r["categories"] or not r["reviewer"].strip():
        raise ValueError("Review must identify reviewer, families and categories")
    decision = c["decision_at"]
    if (
        not (
            0
            <= r["episode_start"]
            <= r["episode_end"]
            <= decision
            <= c["finished_at"]
            <= r["reviewed_at"]
        )
        or not 0 <= r["target_available_at"] <= decision
    ):
        raise ValueError("Review or episode is not causal")
    times, evidence = r["evidence_available_at"], c["packet"]["evidence"]
    if set(times) != set(evidence) or any(not 0 <= t <= decision for t in times.values()):
        raise ValueError("Every evidence handle must have causal source availability")
    if row["role"] != c["role"] or row["prompt"][-1]["content"] != canonical(c["packet"]):
        raise ValueError("Training input differs from its retained original packet")
    if row["completion"][0]["content"] != canonical(provenance["target"]):
        raise ValueError("Training completion differs from the reviewed target")


def import_qtrades(
    source: Path, destination: Path, expected_contract: str, *, intended_splits: tuple = SPLITS[:3]
) -> dict:
    manifest = read_object(source / "manifest.json")
    identity = manifest.pop("corpus_sha256", None)
    if identity != qtrades_digest(manifest) or manifest.get("format") != "qtrades-role-training-v1":
        raise ValueError("Changed or unsupported Q-Trades completion manifest")
    if manifest.get("contract_sha256") != expected_contract:
        raise ValueError("Use the explicitly expected application role contract")
    if not (
        0 <= manifest["train_end"] < manifest["validation_end"]
        and manifest["embargo_seconds"] >= 0
        and isinstance(manifest.get("protected_packets_sha256"), str)
        and len(manifest["protected_packets_sha256"]) == 64
    ):
        raise ValueError("Require ordered cutoffs, embargo and protected evaluation identity")
    names = {s + suffix + ".jsonl" for s in SPLITS[:3] for suffix in ("", ".provenance")}
    if set(manifest.get("files", {})) != names:
        raise ValueError("Unexpected Q-Trades export membership")
    files, metadata, groups, seen, inputs, sealed = {}, {}, {}, set(), {}, {}
    for split in SPLITS[:3]:
        for suffix in ("", ".provenance"):
            name = split + suffix + ".jsonl"
            path, meta = source / name, manifest["files"][name]
            if sha_file(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
                raise ValueError("Changed exported corpus file")
        if split not in intended_splits:
            if not manifest.get("exposure_metadata"):
                raise ValueError("Selective preparation requires complete source exposure metadata")
            sealed[split] = {
                "path": source / (split + ".jsonl"),
                "count": manifest["counts"][split],
            }
            continue
        rows = read_rows(source / (split + ".jsonl"))
        prov = read_rows(source / (split + ".provenance.jsonl"))
        if (
            (not rows and manifest.get("purpose") != "training_preparation")
            or len(rows) != len(prov)
            or len(rows) != manifest["counts"][split]
        ):
            raise ValueError("Corpus counts or provenance alignment changed")
        files[split] = rows
        for row, origin in zip(rows, prov, strict=True):
            check_row(row)
            _review_qtrades(row, origin, expected_contract)
            c, r = origin["candidate"], origin["review"]
            if "source_reviews" in manifest:
                linked = manifest["source_reviews"].get(row["id"], {})
                if (
                    not isinstance(linked.get("revision"), int)
                    or linked["revision"] < 1
                    or linked.get("review_sha256") != qtrades_digest(r)
                    or linked.get("target_sha256") != qtrades_digest(origin["target"])
                ):
                    raise ValueError("Saved review revision differs from imported provenance")
            if row["id"] in seen:
                raise ValueError("Duplicated original candidate")
            seen.add(row["id"])
            content = digest(c["packet"]["evidence"])
            if content in inputs and inputs[content] != split:
                raise ValueError("Evidence crosses splits")
            inputs[content] = split
            for group in c["group_ids"] + ["family:" + f for f in r["family_ids"]]:
                if group in groups and groups[group] != split:
                    raise ValueError("Episode family crosses splits")
                groups[group] = split
            at = c["decision_at"]
            start = r["episode_start"]
            train_end, val_end = manifest["train_end"], manifest["validation_end"]
            embargo = manifest["embargo_seconds"]
            causal_split = (
                "train"
                if at <= train_end
                else "validation"
                if start > train_end + embargo and at <= val_end
                else "test"
                if start > val_end + embargo
                else None
            )
            if split != causal_split:
                raise ValueError("Chronology/embargo differs from the frozen split")
            metadata[row["id"]] = {
                "role": c["role"],
                "families": r["family_ids"],
                "categories": r["categories"],
                "data_basis": r["data_basis"],
            }
    preparation = manifest.get("purpose") == "training_preparation"
    if preparation and (
        not files.get("train") or manifest["counts"]["validation"] or manifest["counts"]["test"]
    ):
        raise ValueError("Preparation-only handoff must contain intended train inputs only")
    return write_corpus(
        files,
        destination,
        {
            "source": "qtrades-reviewed-export",
            "source_corpus_sha256": identity,
            "application_contract_sha256": expected_contract,
            "metadata": metadata,
            "exposure_metadata": manifest.get("exposure_metadata", {}),
            "source_reviews": manifest.get("source_reviews", {}),
            "purpose": manifest.get("purpose", "training_and_evaluation"),
            "regression_available": False,
            "limitations": [
                "Application target validation and semantic review remain authoritative.",
                "No general regression set supplied; production comparison is incomplete.",
            ],
        },
        sealed_files=sealed,
    )


def write_corpus(
    files: dict, destination: Path, extra: dict, *, sealed_files: dict | None = None
) -> dict:
    metadata = extra.get("metadata", {})
    manifest = {"format": "llm-lab-corpus-v1", "files": {}, **extra}
    bodies = {}
    for split, rows in files.items():
        if split not in SPLITS:
            raise ValueError("Unsupported dataset split")
        for row in rows:
            check_row(row)
        bodies[split + ".jsonl"] = "".join(canonical(row) + "\n" for row in rows)
    dest = new_directory(destination)
    for name, body in bodies.items():
        (dest / name).write_text(body, encoding="utf-8", newline="\n")
        split = name.removesuffix(".jsonl")
        manifest["files"][name] = {
            "sha256": sha_file(dest / name),
            "bytes": (dest / name).stat().st_size,
            "count": len(files[split]),
        }
    for split, meta in (sealed_files or {}).items():
        if split not in SPLITS or split in files:
            raise ValueError("Invalid excluded split")
        name = split + ".jsonl"
        # Copy frozen bytes without parsing excluded final-evaluation targets.
        import shutil

        shutil.copyfile(meta["path"], dest / name)
        manifest["files"][name] = {
            "sha256": sha_file(dest / name),
            "bytes": (dest / name).stat().st_size,
            "count": meta["count"],
        }
    manifest["categories"] = dict(
        Counter(category for entry in metadata.values() for category in entry.get("categories", []))
    )
    manifest["corpus_sha256"] = digest(manifest)
    write_new(dest / "manifest.json", manifest)
    return manifest


def verify_corpus(directory: Path, *, splits: tuple = SPLITS) -> tuple[dict, dict]:
    manifest = read_object(directory / "manifest.json")
    body = {k: v for k, v in manifest.items() if k != "corpus_sha256"}
    if manifest.get("format") != "llm-lab-corpus-v1" or digest(body) != manifest["corpus_sha256"]:
        raise ValueError("Corpus completion manifest changed")
    if not {s + ".jsonl" for s in SPLITS[:3]} <= set(manifest["files"]):
        raise ValueError("Require frozen train/validation/test splits")
    files, seen, prompts = {}, set(), {}
    for name, meta in manifest["files"].items():
        if name not in {s + ".jsonl" for s in SPLITS}:
            raise ValueError("Unexpected corpus file")
        path = directory / name
        if sha_file(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise ValueError("Corpus content changed")
        split = name.removesuffix(".jsonl")
        # Verify all immutable files, but do not parse excluded evaluation targets.
        if split not in splits:
            continue
        rows = read_rows(path)
        if (not rows and manifest.get("purpose") != "training_preparation") or len(rows) != meta[
            "count"
        ]:
            raise ValueError("Corpus count changed or empty split")
        split = name.removesuffix(".jsonl")
        for row in rows:
            check_row(row)
            if row["id"] in seen:
                raise ValueError("Example identity crosses or repeats within splits")
            seen.add(row["id"])
            content = digest(row["prompt"])
            if content in prompts and prompts[content] != split:
                raise ValueError("Identical prompt crosses splits")
            prompts[content] = split
        files[split] = rows
    return manifest, files
