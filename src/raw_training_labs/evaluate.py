"""Model-appropriate matched generation and complete task-quality denominators."""

import time

import psutil
from llm_lab.config import Recipe
from llm_lab.io import digest
from llm_lab.masking import token_ids
from llm_lab.model import load_base, tokenizer_at

from raw_training_labs.scoring import aggregate, score


def arm(base, recipe: Recipe, profile, files, criteria, *, adapter=None, progress=None, stop=None):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    torch.set_num_threads(recipe.cpu_threads)
    tokenizer = tokenizer_at(base)
    model, loading_error = None, None
    started = time.monotonic()
    try:
        model = load_base(base, recipe)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, adapter, is_trainable=False, local_files_only=True, use_safetensors=True
            )
        model.eval()
        model.config.use_cache = True
    except Exception as exc:
        loading_error = f"{type(exc).__name__}: {exc}"
    load_seconds = time.monotonic() - started
    cases = []
    settings = profile["generation"]
    for split, rows in files.items():
        for row in rows:
            if stop and stop():
                raise RuntimeError("Evaluation cancelled; partial cases retained in progress")
            elapsed_start = time.monotonic()
            raw, status, tokens, error = "", "ok", 0, None
            ids = token_ids(
                tokenizer.apply_chat_template(
                    row["prompt"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=profile["enable_thinking"],
                )
            )
            seed = int(digest({"id": row["id"], "seed": recipe.seed})[:8], 16) % (2**31)
            try:
                if loading_error:
                    raise RuntimeError(loading_error)
                if len(ids) + settings["max_new_tokens"] > profile["context"]:
                    raise ValueError(
                        "Input plus generation exceeds approved context; no truncation"
                    )

                class Stop(StoppingCriteria):
                    def __init__(self, call_started):
                        self.call_started = call_started

                    def __call__(self, input_ids, scores, **kwargs):
                        return time.monotonic() - self.call_started >= settings[
                            "max_seconds"
                        ] or bool(stop and stop())

                torch.manual_seed(seed)
                input_ids = torch.tensor([ids], device=recipe.device)
                options = {
                    "max_new_tokens": settings["max_new_tokens"],
                    "do_sample": settings["do_sample"],
                    "max_time": settings["max_seconds"],
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "stopping_criteria": StoppingCriteriaList([Stop(elapsed_start)]),
                }
                if settings["do_sample"]:
                    options.update({k: settings[k] for k in ("temperature", "top_p", "top_k")})
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **options
                    )
                generated = output[0, len(ids) :].tolist()
                tokens = len(generated)
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                if time.monotonic() - elapsed_start >= settings["max_seconds"]:
                    status = "timed_out"
                if stop and stop():
                    status = "cancelled"
                if "<think>" in raw or "</think>" in raw:
                    status = "unexpected_thinking_output"
            except Exception as exc:
                status, error = "failed", f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - elapsed_start
            case = {
                "id": row["id"],
                "split": split,
                "input": row["prompt"],
                "input_sha256": digest(row["prompt"]),
                "input_tokens": len(ids),
                "expected": row["completion"][0]["content"],
                "raw": raw,
                "seed": seed,
                "output_tokens": tokens,
                "token_budget_reached": tokens >= settings["max_new_tokens"],
                "status": status,
                "error": error,
                "latency_seconds": elapsed,
                "rss_bytes": psutil.Process().memory_info().rss,
            }
            case["score"] = score(raw, case["expected"], criteria, call_status=status)
            cases.append(case)
            if progress:
                progress(cases)
    del model
    import gc

    gc.collect()
    return {
        "cases": cases,
        "metrics": aggregate(cases),
        "load_seconds": load_seconds,
        "generation": settings,
        "thinking": profile["enable_thinking"],
        "recipe": recipe.model_dump(),
        "loading_error": loading_error,
    }
