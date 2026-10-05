"""Standalone PEFT package verifier. No RAW or original Lab imports required."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Only regular package files are accepted")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify(package, base):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads((package / "package.json").read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        target = package / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not target.resolve().is_relative_to(package.resolve())
        ):
            raise ValueError("Escaping package file")
        if sha(target) != identity:
            raise ValueError("Delivery package content changed")
    expected = manifest["base_manifest"]
    actual = json.loads((base / "base-manifest.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("Runtime base identity differs from the exported exact base")
    for name, identity in expected["files"].items():
        if Path(name).name != name or sha(base / name) != identity:
            raise ValueError("Runtime base/tokenizer changed")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        package / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    base_model, loading = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(
        loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Export runtime did not completely load the pretrained base")
    model = PeftModel.from_pretrained(
        base_model,
        package / "adapter",
        is_trainable=False,
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()
    inputs = tokenizer.apply_chat_template(
        manifest["probe"]["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    settings = manifest["profile"]["generation"]
    torch.manual_seed(manifest["probe"]["seed"])
    started = time.monotonic()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            use_cache=True,
            do_sample=settings["do_sample"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            max_new_tokens=settings["max_new_tokens"],
            max_time=settings["max_seconds"],
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    if raw != manifest["probe"]["raw"]:
        raise ValueError("Standalone runtime output differs from retained candidate probe")
    return {
        "status": "verified",
        "runtime": "Independent Transformers/PEFT process",
        "base_sha256": actual["base_sha256"],
        "raw": raw,
        "latency_seconds": time.monotonic() - started,
        "no_lab_imports": True,
        "quality_acceptance": manifest["quality_acceptance"],
        "activated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(verify(args.package, args.base)))
