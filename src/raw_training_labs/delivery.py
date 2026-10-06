"""Sensitive PEFT packages with exact identities and independently checked runtime."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from llm_lab.io import read_object, sha_file, write_new
from llm_lab.model import environment, verify_model

from raw_training_labs.jobs import completed


def export(service, project_id, request, directory, profile):
    store = service.store
    corpus = store.artifact(project_id, request["corpus_id"], "corpus")
    manifest = read_object(corpus["path"] / "manifest.json")
    if any(not r["review"]["rights"]["export"] for r in manifest["source_reviews"].values()):
        raise ValueError(
            "Export permission is missing for a reviewed example; sensitive adapter export blocked"
        )
    candidate = completed(store, project_id, request["candidate_id"], "train")
    run = store.artifact_path(project_id, f"jobs/{candidate['id']}/run")
    receipt = read_object(run / "run.json")
    for name, identity in receipt["candidate_files"].items():
        if Path(name).name != name or sha_file(run / "best-adapter" / name) != identity:
            raise ValueError("Selected candidate adapter changed")
    with store.connect() as db:
        rows = db.execute(
            "SELECT id,request FROM jobs WHERE project=? AND kind='compare' "
            "AND status='completed' ORDER BY created DESC",
            (project_id,),
        ).fetchall()
    comparison_id = next(
        r["id"]
        for r in rows
        if json.loads(r["request"])["candidate_id"] == candidate["id"]
        and json.loads(r["request"])["baseline_id"] == request["baseline_id"]
    )
    compare_root = store.artifact_path(project_id, f"jobs/{comparison_id}")
    comparison = read_object(compare_root / "result.json")
    probe = comparison["comparison"]["pairs"][0]["candidate"]
    if probe["status"] != "ok":
        raise ValueError(
            "Select a successfully executed candidate comparison before runtime verification"
        )
    base = store.artifact_path(project_id, f"models/{profile['id']}")
    frozen_base = verify_model(base)
    if (
        receipt["base_sha256"] != frozen_base["base_sha256"]
        or receipt["corpus_sha256"] != manifest["corpus_sha256"]
        or receipt["environment"] != environment()
    ):
        raise ValueError("Package requires the exact retained candidate base/data/environment")
    package = directory / "package"
    package.mkdir(exist_ok=False)
    shutil.copytree(run / "best-adapter", package / "adapter")
    (package / "tokenizer").mkdir()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        if (base / name).is_file():
            shutil.copyfile(base / name, package / "tokenizer" / name)
    shutil.copyfile(base / "LICENSE", package / "BASE-LICENSE")
    shutil.copyfile(Path(__file__).with_name("runtime_template.py"), package / "verify_runtime.py")
    write_new(package / "recipe.json", receipt["recipe"])
    write_new(package / "evaluation.json", comparison)
    (package / "README.txt").write_text(
        "RAW Training Labs: sensitive development PEFT package.\n"
        "Exact base must be independently supplied under its license; "
        "base weights are not bundled.\n"
        "Install the recorded dependencies; run verify_runtime.py "
        "--package <folder> --base <exact base>.\n"
        "Tokenizer/template and non-thinking generation settings are required.\n"
        "Do not share without permission. Adapter weights may retain source information.\n"
        "Runtime verification does not approve task quality, customer use, "
        "installation or activation.\n",
        encoding="utf-8",
    )
    files = {
        str(p.relative_to(package)).replace("\\", "/"): sha_file(p)
        for p in sorted(package.rglob("*"))
        if p.is_file()
    }
    write_new(
        package / "package.json",
        {
            "format": "raw-peft-package-v1",
            "project_id": project_id,
            "candidate_id": candidate["id"],
            "comparison_id": comparison_id,
            "files": files,
            "base_manifest": frozen_base,
            "profile": profile,
            "environment": environment(),
            "quality_acceptance": comparison["comparison"]["criteria_passed"],
            "probe": {"prompt": probe["input"], "seed": probe["seed"], "raw": probe["raw"]},
            "limitations": [
                "Synthetic demonstration only; independent customer acceptance is outstanding.",
                "PEFT adapter requires its exact base, tokenizer, template and runtime.",
                "No GGUF/Ollama conversion, automatic installation, activation "
                "or model unlearning claim.",
            ],
        },
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(package / "verify_runtime.py"),
            "--package",
            str(package),
            "--base",
            str(base),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=package,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    (directory / "runtime.log").write_text(result.stderr[-64000:], encoding="utf-8")
    if result.returncode:
        raise ValueError("Independent exported runtime failed; inspect retained runtime.log")
    verification = json.loads(result.stdout.strip().splitlines()[-1])
    write_new(directory / "runtime-verification.json", verification)
    return {
        "package": str(package),
        "package_sha256": sha_file(package / "package.json"),
        "runtime_verification": verification,
        "activated": False,
        "stage": "runtime_verified_development_package",
    }
