"""Load only declared local safe checkpoints; retain their exact source hashes."""

import importlib.metadata
import platform
from pathlib import Path

from llm_lab.io import digest, private_path, read_object, sha_file, write_new


def environment():
    names = ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors")
    return {name: importlib.metadata.version(name) for name in names} | {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bitsandbytes": optional_version("bitsandbytes"),
    }


def optional_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def source_identity():
    directory = Path(__file__).parent
    files = {p.name: sha_file(p) for p in sorted(directory.glob("*.py"))}
    return {"files": files, "source_sha256": digest(files)}


def freeze_model(directory: Path, *, provenance: str, revision: str, rights: str) -> dict:
    directory = private_path(directory)
    if (directory / "base-manifest.json").exists():
        raise ValueError("Model source is already frozen; verify instead of replacing its identity")
    if not provenance.strip() or not revision.strip() or not rights.strip():
        raise ValueError("Declare model source, exact revision and applicable rights")
    names = {p.name for p in directory.iterdir() if p.is_file() and p.name != "base-manifest.json"}
    if "config.json" not in names or "tokenizer.json" not in names:
        raise ValueError("Require local model config and exact tokenizer")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Require safetensors weights; pickle checkpoints are not accepted")
    if any(name.endswith((".bin", ".pt", ".py")) for name in names):
        raise ValueError("No executable custom model code or pickle model weights")
    manifest = {
        "format": "llm-lab-base-v1",
        "provenance": provenance,
        "revision": revision,
        "rights": rights,
        "files": {name: sha_file(directory / name) for name in sorted(names)},
    }
    manifest["base_sha256"] = digest(manifest)
    write_new(directory / "base-manifest.json", manifest)
    return manifest


def verify_model(directory: Path):
    manifest = read_object(directory / "base-manifest.json")
    body = {k: v for k, v in manifest.items() if k != "base_sha256"}
    if manifest.get("format") != "llm-lab-base-v1" or digest(body) != manifest["base_sha256"]:
        raise ValueError("Base model manifest changed")
    names = {p.name for p in directory.iterdir() if p.is_file() and p.name != "base-manifest.json"}
    if names != set(manifest["files"]):
        raise ValueError("Base checkpoint membership changed")
    for name, identity in manifest["files"].items():
        if Path(name).name != name or sha_file(directory / name) != identity:
            raise ValueError("Base checkpoint or tokenizer changed")
    return manifest


def tokenizer_at(directory: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a padding nor an end token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(directory: Path, recipe):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen3_5ForCausalLM

    cls = Qwen3_5ForCausalLM if recipe.model_kind == "qwen3_5_text" else AutoModelForCausalLM
    dtype = getattr(torch, recipe.precision)
    options = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    if recipe.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("This environment has no working CUDA; no CPU fallback is implied")
    if recipe.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support the declared bfloat16 profile")
    if recipe.quantization == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        options["device_map"] = {"": 0}
    model, loading = cls.from_pretrained(directory, output_loading_info=True, **options)
    unexpected = loading.get("unexpected_keys", [])
    allowed_vision = recipe.model_kind == "qwen3_5_text" and all(
        name.startswith("model.visual.") for name in unexpected
    )
    if (
        loading.get("missing_keys")
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
        or (unexpected and not allowed_vision)
    ):
        raise ValueError(
            "Pretrained text weights did not load completely; random initialization is refused"
        )
    model._lab_loading_info = {
        "missing_keys": sorted(loading.get("missing_keys", [])),
        "unexpected_vision_keys": len(unexpected),
        "unexpected_keys_sha256": digest(sorted(unexpected)),
        "text_weights_complete": True,
    }
    if recipe.quantization == "none":
        model.to(recipe.device)
    else:
        from peft import prepare_model_for_kbit_training

        # Use the same promoted non-quantized layers for the initial baseline,
        # adapter optimization, and both later evaluation arms.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
    return model


def attach_adapter(model, recipe):
    from peft import LoraConfig, get_peft_model

    targets = "all-linear" if recipe.targets == ["all-linear"] else recipe.targets
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe.rank,
            lora_alpha=recipe.alpha,
            lora_dropout=recipe.dropout,
            target_modules=targets,
            task_type="CAUSAL_LM",
            use_rslora=recipe.rank_stabilized,
            bias="none",
        ),
    )
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters or any("lora_" not in name for name, _ in parameters):
        raise ValueError("Only verified LoRA parameters may be optimized in this recipe")
    if recipe.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, {
        "trainable_parameters": sum(p.numel() for _, p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_names": [name for name, _ in parameters],
    }
