"""Explicit independent acquisition of the single approved pinned demo checkpoint."""

from pathlib import Path

from llm_lab.io import read_object, write_new
from llm_lab.model import freeze_model, verify_model

from raw_training_labs.profiles import supported


def acquire(service, project_id, profile_id, *, authorized=False):
    if not authorized:
        raise ValueError("Model download requires explicit authorization")
    profile = supported(profile_id)
    service.store.project(project_id)
    base = service.store.artifact_path(project_id, f"models/{profile_id}")
    if (base / "base-manifest.json").exists():
        return verify_model(base)
    service.store.budget(reserve=3 * 1024**3)
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi(token=False).model_info(
        profile["model"], revision=profile["revision"], files_metadata=True
    )
    if info.sha != profile["revision"]:
        raise ValueError("Upstream revision differs from the pinned model identity")
    allowed = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "model.safetensors.index.json",
        "LICENSE",
        "README.md",
    ]
    selected = [s for s in info.siblings if s.rfilename in allowed]
    if sum(s.size or 0 for s in selected) > 2 * 1024**3:
        raise ValueError("Acquisition exceeds the approved small-model budget")
    base.parent.mkdir(parents=True, exist_ok=True)
    # resumable transfer belongs only to this explicitly requested project/model.
    snapshot_download(
        profile["model"],
        revision=profile["revision"],
        local_dir=base,
        allow_patterns=allowed,
        max_workers=2,
        token=False,
    )
    config = read_object(base / "config.json")
    if config.get("model_type") != "qwen3" or not (base / "LICENSE").is_file():
        raise ValueError("Expected Qwen3 architecture and upstream license")
    result = freeze_model(
        base,
        provenance=profile["model"],
        revision=profile["revision"],
        rights="Apache-2.0; retain LICENSE and attribution",
    )
    service.store.budget()
    receipt = base.parent / f"{profile_id}-acquisition.json"
    if not receipt.exists():
        write_new(
            receipt,
            {
                "model": profile["model"],
                "revision": profile["revision"],
                "base_sha256": result["base_sha256"],
                "independent_acquisition": True,
                "original_model_reused": False,
                "directory": str(Path(base).resolve()),
            },
        )
    return result
