"""Read existing frozen manifests, never evaluation answers or a second split database."""

from pathlib import Path

from llm_lab.io import digest, read_object, sha_file, write_new


def safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ":" in relative:
        raise ValueError("Use a relative private artifact identity")
    path = root / relative
    if ".." in Path(relative).parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Linked or escaping artifact path refused")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact is outside the approved private root")
    return path


def related(a: dict, b: dict) -> bool:
    if a.get("evidence_key") and a.get("evidence_key") == b.get("evidence_key"):
        return True
    if set(a.get("families", [])) & set(b.get("families", [])):
        return True
    x, y = a.get("window"), b.get("window")
    return bool(x and y and max(x[0], y[0]) <= min(x[1], y[1]))


def check(root: Path, policy: dict, incoming: list[dict]) -> dict:
    if len(incoming) > 64 or len(policy.get("studies", [])) > 64:
        raise ValueError("Exposure check exceeds the bounded study/example limit")
    problems, sources, limitations = [], [], []
    studies = list(policy.get("studies", []))
    # Existing accepted handoff manifests are exposure history; no second split index.
    jobs = root / "qtrades-handoffs"
    if jobs.exists():
        entries = list(jobs.iterdir())
        if len(entries) > 512:
            raise ValueError("Handoff history exceeds bounded inspection; archive via Lab policy")
        for job in entries:
            if (job / "receipt.json").is_file() and (job / "corpus/manifest.json").is_file():
                receipt = read_object(job / "receipt.json")
                if receipt.get("stage") == "prepared":
                    path = job / "corpus/manifest.json"
                    studies.append(
                        {
                            "id": receipt["study_id"],
                            "sealed": False,
                            "manifest": str(path.relative_to(root)),
                            "sha256": sha_file(path),
                        }
                    )
    for study in studies:
        manifest_path = safe_file(root, study["manifest"])
        if sha_file(manifest_path) != study["sha256"]:
            raise ValueError(
                "Frozen study manifest changed; update verified configuration explicitly"
            )
        manifest = read_object(manifest_path)
        history = manifest.get("exposure_metadata", manifest.get("metadata", {}))
        if not history:
            raise ValueError("Study has no exposure metadata; independence is unresolved")
        retired = set()
        for retirement in policy.get("retirements", []):
            record = read_object(safe_file(root, retirement))
            if record.get("manifest_sha256") == study["sha256"]:
                if study.get("sealed"):
                    raise ValueError("Sealed material cannot be retired through this handoff")
                retired.update(record["splits"])
        legacy = "exposure_metadata" not in manifest
        if legacy:
            limitations.append(
                "Legacy manifest lacks per-case split/window/content keys: all recorded families "
                "are conservatively protected; unseen source relationships remain unproved."
            )
        for old_id, old in history.items():
            prior = dict(old)
            prior["families"] = old.get("families", []) + [
                "family:" + f for f in old.get("families", [])
            ]
            split = prior.get("split", "unknown_legacy")
            protected = split in {"test", "validation", "unknown_legacy"} and split not in retired
            for new in incoming:
                # Retiring old evaluation allows training/regression reuse, never fresh evaluation.
                evaluation_reuse = new.get("split") in {"validation", "test"}
                if (protected or evaluation_reuse) and related(new, prior):
                    problems.append(
                        {
                            "id": new["id"],
                            "code": "cross_study_exposure",
                            "study": study["id"],
                            "related_id": old_id,
                            "reason": "Related protected evaluation family/window/content",
                        }
                    )
        sources.append(
            {
                "study": study["id"],
                "manifest_sha256": study["sha256"],
                "sealed": bool(study.get("sealed")),
                "retired_splits": sorted(retired),
            }
        )
    return {
        "eligible": not problems,
        "problems": problems,
        "sources": sources,
        "policy_sha256": digest(policy),
        "limitations": sorted(set(limitations)),
    }


def retire(root: Path, policy: dict, request: dict, destination: Path) -> dict:
    study = next((s for s in policy["studies"] if s["id"] == request["study"]), None)
    if not study or study.get("sealed"):
        raise ValueError("Unknown or sealed evaluation study cannot be retired")
    if not request.get("reviewer") or len(request.get("reason", "")) < 20:
        raise ValueError("Retirement requires explicit identity and a specific reason")
    if request.get("splits") != ["validation", "test", "unknown_legacy"]:
        raise ValueError("Retire the explicitly selected study evaluation population")
    if sha_file(safe_file(root, study["manifest"])) != study["sha256"]:
        raise ValueError("Study manifest changed")
    result = {
        "format": "llm-lab-evaluation-retirement-v1",
        **request,
        "manifest_sha256": study["sha256"],
        "freshness": "retired_into_training_regression",
        "original_study_unchanged": True,
    }
    write_new(destination, result)
    return result
