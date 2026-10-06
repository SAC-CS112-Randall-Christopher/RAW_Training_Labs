import os

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def hybrid(tmp_path_factory):
    from llm_lab.fixtures import create_fixture

    return create_fixture(tmp_path_factory.mktemp("hybrid-parent") / "fixture", steps=6)


@pytest.fixture
def source_export(tmp_path):
    """Authored Q-Trades wire-format fixture, never a real reviewed account export."""
    from llm_lab.io import canonical, qtrades_digest, sha_file, write_new

    source = tmp_path / "source"
    source.mkdir()
    contract = "a" * 64
    manifest = {
        "format": "qtrades-role-training-v1",
        "contract_sha256": contract,
        "train_end": 150.0,
        "validation_end": 250.0,
        "embargo_seconds": 10.0,
        "protected_packets_sha256": "b" * 64,
        "counts": {},
        "files": {},
    }
    for split, at in (("train", 100.0), ("validation", 200.0), ("test", 300.0)):
        packet = {"evidence": {"e0": {"fact": split}}, "question": split, "capabilities": {}}
        target = {"action": "no_change", "evidence_ids": ["e0"]}
        candidate = {
            "format": manifest["format"],
            "contract_sha256": contract,
            "role": "researcher",
            "stage": "idea",
            "task_id": split,
            "attempt": 1,
            "decision_at": at,
            "finished_at": at + 1,
            "group_ids": ["task:" + split],
            "packet": packet,
            "original_answer": target,
            "packet_sha256": qtrades_digest(packet),
        }
        candidate["candidate_sha256"] = qtrades_digest(candidate)
        origin = {
            "candidate": candidate,
            "target": target,
            "review": {
                "approved": True,
                "rights_confirmed": True,
                "reviewer": "Procedural test author",
                "reviewer_kind": "human",
                "reviewer_authored_material": True,
                "claim_scope": "interpretation",
                "reviewed_at": at + 2,
                "rationale": "Authored wire-format test, not actual operator adjudication.",
                "data_basis": "synthetic",
                "family_ids": [split],
                "categories": ["workflow"],
                "episode_start": at - 5,
                "episode_end": at - 1,
                "target_available_at": at - 1,
                "evidence_available_at": {"e0": at - 1},
            },
        }
        row = {
            "id": candidate["candidate_sha256"],
            "role": "researcher",
            "prompt": [
                {"role": "system", "content": "Procedural source contract."},
                {"role": "user", "content": canonical(packet)},
            ],
            "completion": [{"role": "assistant", "content": canonical(target)}],
        }
        for suffix, content in (("", row), (".provenance", origin)):
            name = split + suffix + ".jsonl"
            (source / name).write_text(canonical(content) + "\n", encoding="utf-8")
            manifest["files"][name] = {
                "sha256": sha_file(source / name),
                "bytes": (source / name).stat().st_size,
            }
        manifest["counts"][split] = 1
    manifest["corpus_sha256"] = qtrades_digest(manifest)
    write_new(source / "manifest.json", manifest)
    return source, contract


def reseal_export(source):
    from llm_lab.io import qtrades_digest, read_object, sha_file, write_state

    manifest = read_object(source / "manifest.json")
    manifest.pop("corpus_sha256")
    for name, meta in manifest["files"].items():
        meta.update(sha256=sha_file(source / name), bytes=(source / name).stat().st_size)
    manifest["corpus_sha256"] = qtrades_digest(manifest)
    write_state(source / "manifest.json", manifest)
