"""Exact conversation rendering, explicit completion labels, and no truncation."""

from collections.abc import Mapping

from llm_lab.io import digest


def token_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected exactly one rendered conversation")
        value = value[0]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("Renderer did not return a single token sequence")
    return value


def encode(row, tokenizer, max_length, *, enable_thinking=False):
    full = tokenizer.apply_chat_template(
        row["prompt"] + row["completion"],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    prefix = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # A template can return a BatchEncoding in newer Transformers.
    full, prefix = token_ids(full), token_ids(prefix)
    if full[: len(prefix)] != prefix:
        raise ValueError("Chat template is not prefix-preserving; verify a compatible renderer")
    if len(full) > max_length:
        raise ValueError(f"Example {row['id']} exceeds context budget; no silent truncation")
    if len(full) <= len(prefix):
        raise ValueError("No supervised completion tokens after rendering")
    return {
        "id": row["id"],
        "input_ids": list(full),
        "labels": [-100] * len(prefix) + list(full[len(prefix) :]),
        "prompt_tokens": len(prefix),
        "completion_tokens": len(full) - len(prefix),
    }


def preflight(files, tokenizer, max_length, *, enable_thinking=False):
    template = tokenizer.chat_template
    if not template or tokenizer.pad_token_id is None:
        raise ValueError("Require the actual chat template and a declared padding token")
    encoded, receipt = (
        {},
        {
            "template_sha256": digest(template),
            "renderer_kwargs": {"enable_thinking": enable_thinking},
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "splits": {},
        },
    )
    for split, rows in files.items():
        values = [
            encode(row, tokenizer, max_length, enable_thinking=enable_thinking) for row in rows
        ]
        encoded[split] = values
        receipt["splits"][split] = {
            "examples": len(values),
            "max_tokens": max(len(v["input_ids"]) for v in values),
            "input_tokens": sum(v["prompt_tokens"] for v in values),
            "supervised_tokens": sum(v["completion_tokens"] for v in values),
            "encoded_sha256": digest(values),
        }
    receipt["masking"] = "Completion only; input and padding labels are -100; no truncation"
    return receipt, encoded


def collate(rows, tokenizer, device):
    import torch

    width = max(len(row["input_ids"]) for row in rows)
    ids, labels, masks = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(masks, device=device),
    }
