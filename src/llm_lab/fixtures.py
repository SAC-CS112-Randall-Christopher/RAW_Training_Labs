"""Locally constructed, randomly initialized hybrid model; software proof only."""

import string
from pathlib import Path

from llm_lab.config import Recipe
from llm_lab.corpus import write_corpus
from llm_lab.io import canonical, new_directory, write_new
from llm_lab.model import freeze_model


def fixture_files():
    cases = [
        ("Fees exceed gains; gross 3, fees 5.", "reject"),
        ("Price source is missing; execution cannot be checked.", "wait"),
        ("New supported comparison; same costs, uncertain outcome.", "propose_experiment"),
        ("No independent information since the prior experiment.", "no_change"),
    ]
    files, metadata = {}, {}
    for split, prefix in (
        ("train", "Alpha"),
        ("validation", "Beta"),
        ("test", "Gamma"),
        ("regression", "General"),
    ):
        rows = []
        for index, (fact, action) in enumerate(cases):
            identity = f"procedural-{split}-{index}"
            question = (
                f"{prefix} case {index}. {fact}"
                if split != "regression"
                else f"{prefix} task {index}. Preserve units and report missing evidence."
            )
            target = {"action": action if split != "regression" else "wait"}
            rows.append(
                {
                    "id": identity,
                    "role": "researcher" if index % 2 else "reviewer",
                    "prompt": [
                        {"role": "system", "content": "Return the supported JSON action."},
                        {"role": "user", "content": question},
                    ],
                    "completion": [{"role": "assistant", "content": canonical(target)}],
                }
            )
            metadata[identity] = {
                "families": [identity],
                "categories": ["procedural"],
                "data_basis": "synthetic",
                "review": "Authored checkable fixture",
            }
        files[split] = rows
    return files, metadata


def create_fixture(root: Path, *, device="cpu", steps=12, quantization="none"):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    root = new_directory(root)
    files, metadata = fixture_files()
    write_corpus(
        files,
        root / "corpus",
        {
            "source": "locally-authored-procedural-fixture",
            "metadata": metadata,
            "regression_available": True,
            "limitations": [
                "Synthetic procedural software verification; not market evidence.",
                "No pretrained knowledge or actual Qwen3.5-4B weights are present.",
            ],
        },
    )
    base = new_directory(root / "base")
    specials = ["[PAD]", "[UNK]", "[EOS]", "[SYSTEM]", "[USER]", "[ASSISTANT]", "[TOOL]"]
    vocabulary = {
        value: index for index, value in enumerate(specials + list(dict.fromkeys(string.printable)))
    }
    native = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    native.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=specials[3:],
        model_max_length=384,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '[' + message['role']|upper + ']' }}"
        "{{ message['content'] }}{{ eos_token }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '[ASSISTANT]' }}{% endif %}"
    )
    tokenizer.save_pretrained(base)
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=384,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )
    Qwen3_5ForCausalLM(config).save_pretrained(base, safe_serialization=True)
    freeze_model(
        base,
        provenance="Locally initialized tiny Qwen3.5 hybrid architecture fixture",
        revision="fixture-v1-seed17",
        rights="Locally authored procedural test",
    )
    recipe = Recipe(
        name="hybrid-procedural-smoke",
        model_kind="qwen3_5_text",
        device=device,
        precision="float16" if quantization == "nf4" else "float32",
        quantization=quantization,
        rank=4,
        alpha=8,
        dropout=0.0,
        learning_rate=0.003,
        steps=steps,
        warmup_steps=0,
        eval_every=max(1, steps // 3),
        accumulation=2,
        max_length=384,
        max_run_seconds=180,
        gradient_checkpointing=quantization == "nf4",
    )
    write_new(root / "recipe.json", recipe.model_dump())
    return root, recipe
