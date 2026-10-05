"""Matched baseline/candidate diagnostics. Failures remain requested observations."""

import gc
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

from llm_lab.config import Recipe
from llm_lab.corpus import verify_corpus
from llm_lab.io import canonical, digest, new_directory, read_object, sha_file, write_new
from llm_lab.masking import preflight, token_ids
from llm_lab.model import load_base, tokenizer_at, verify_model
from llm_lab.resources import gpu_gate, low_priority
from llm_lab.training import YieldRequested, _context, loss_on, training_gate


def diagnose(text: str, expected: dict, packet: dict | None = None) -> dict:
    result = {
        "json_valid": False,
        "action_match": False,
        "exact_match": False,
        "critical": False,
        "false_rejection": False,
    }
    try:
        answer = json.loads(text)
        if not isinstance(answer, dict):
            return result
        canonical(answer)
    except (ValueError, TypeError):
        return result
    result["json_valid"] = True
    result["exact_match"] = answer == expected
    result["action_match"] = answer.get("action") == expected.get("action")
    allowed = {"approve", "exploratory_paper_only", "propose_experiment"}
    result["critical"] = bool(
        set(answer) & {"funding", "starting_cash", "risk", "sql", "code"}
        or (answer.get("action") in allowed and expected.get("action") not in allowed)
    )
    result["false_rejection"] = expected.get("action") in allowed and answer.get("action") in {
        "reject",
        "no_change",
        "data_wait",
        "wait",
    }
    if packet and isinstance(packet.get("evidence"), dict):
        ids = answer.get("evidence_ids", [])
        if isinstance(ids, list) and any(
            isinstance(value, str) and value not in packet["evidence"] for value in ids
        ):
            result["critical"] = True
    return result


def summarize(rows: list[dict]) -> dict:
    result = {}
    arms = (
        ("baseline", "reference", "candidate")
        if any(row["arm"] == "reference" for row in rows)
        else ("baseline", "candidate")
    )
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        result[arm] = {
            "requested": len(selected),
            "complete": sum(r["status"] == "complete" for r in selected),
            "errors": sum(r["status"] == "error" for r in selected),
            "resource_waits": sum(r["status"] == "resource_wait" for r in selected),
            "timeouts": sum(r["status"] == "timeout" for r in selected),
            **{
                key: sum(bool(r.get(key)) for r in selected)
                for key in (
                    "json_valid",
                    "action_match",
                    "exact_match",
                    "critical",
                    "false_rejection",
                )
            },
            "measured_seconds": sum(r.get("wall_seconds") or 0 for r in selected),
            "measured_output_tokens": sum(r.get("output_tokens") or 0 for r in selected),
        }
    return result


def evaluate(
    run: Path,
    destination: Path,
    *,
    split="validation",
    seeds=(92811,),
    max_new_tokens=256,
    timeout_seconds=60,
    unlock_test=False,
    yield_file=None,
    evaluation_corpus=None,
    reference_run=None,
) -> dict:
    import psutil
    import torch
    from peft import PeftModel
    from transformers import StoppingCriteria, StoppingCriteriaList

    receipt = read_object(run / "run.json")
    if receipt["status"] != "trained":
        raise ValueError("Only a completed training candidate can enter paired evaluation")
    if split not in {"validation", "test", "regression"}:
        raise ValueError("Training examples cannot become evaluation cases")
    if split == "test" and not unlock_test:
        raise ValueError("Explicit --unlock-test records consumption of the reserved test split")
    if not seeds or len(set(seeds)) != len(seeds) or max_new_tokens < 1 or timeout_seconds <= 0:
        raise ValueError("Declare distinct seeds and positive output/time budgets")
    for name, identity in receipt["candidate_files"].items():
        if Path(name).name != name or sha_file(run / "best-adapter" / name) != identity:
            raise ValueError("Selected candidate changed after training")
    recipe = Recipe.model_validate(receipt["recipe"])
    base, corpus = Path(receipt["base_directory"]), Path(receipt["corpus_directory"])
    base_manifest = verify_model(base)
    manifest, files = verify_corpus(corpus, splits=(split,))
    if (
        base_manifest["base_sha256"] != receipt["base_sha256"]
        or manifest["corpus_sha256"] != receipt["corpus_sha256"]
    ):
        raise ValueError("Baseline or corpus changed since training")
    training_corpus_sha256 = manifest["corpus_sha256"]
    reference = None
    if reference_run is not None:
        reference_run = Path(reference_run)
        reference = read_object(reference_run / "run.json")
        if reference["status"] != "trained" or reference_run.resolve() == run.resolve():
            raise ValueError("Reference must be a separate completed frozen training run")
        reference_base = verify_model(Path(reference["base_directory"]))
        reference_corpus, _ = verify_corpus(Path(reference["corpus_directory"]), splits=())
        if (
            reference["base_sha256"] != receipt["base_sha256"]
            or reference_base["base_sha256"] != receipt["base_sha256"]
            or reference_corpus["corpus_sha256"] != reference["corpus_sha256"]
        ):
            raise ValueError("Reference base or original training corpus changed or differs")
        reference_recipe = Recipe.model_validate(reference["recipe"])
        for field in ("model_kind", "device", "precision", "quantization", "enable_thinking"):
            if getattr(reference_recipe, field) != getattr(recipe, field):
                raise ValueError("Reference requires the same base loading and thinking profile")
        for name, identity in reference["candidate_files"].items():
            if (
                Path(name).name != name
                or sha_file(reference_run / "best-adapter" / name) != identity
            ):
                raise ValueError("Frozen reference candidate changed after training")
    if evaluation_corpus is not None:
        if split != "test":
            raise ValueError("An amended corpus is only for the explicitly reserved test split")
        amended, amended_files = verify_corpus(Path(evaluation_corpus), splits=(split,))
        for name in ("train.jsonl", "validation.jsonl"):
            if amended["files"][name] != manifest["files"][name]:
                raise ValueError("Amended evaluation cannot change training or validation bytes")
        manifest, files = amended, amended_files
    if split not in files:
        raise ValueError("No independent regression dataset supplied")
    tokenizer = tokenizer_at(base)
    _, encoded = preflight(
        files, tokenizer, recipe.max_length, enable_thinking=recipe.enable_thinking
    )
    output = new_directory(destination)
    settings = {
        "arms": ["baseline", "reference", "candidate"] if reference else ["baseline", "candidate"],
        "split": split,
        "seeds": list(seeds),
        "max_new_tokens": max_new_tokens,
        "timeout_seconds": timeout_seconds,
        "do_sample": False,
        "corpus_sha256": manifest["corpus_sha256"],
        "training_corpus_sha256": training_corpus_sha256,
        "evaluation_corpus_directory": str(evaluation_corpus or corpus),
        "candidate_profile_sha256": receipt["profile_sha256"],
        "test_consumed": split == "test",
        "enable_thinking": recipe.enable_thinking,
        "stop_token_ids": [tokenizer.eos_token_id],
        "template_sha256": digest(tokenizer.chat_template),
        "retries_per_case": 0,
    }
    if reference:
        settings.update(
            reference_profile_sha256=reference["profile_sha256"],
            reference_run_sha256=sha_file(reference_run / "run.json"),
            reference_candidate_files=reference["candidate_files"],
            reference_training_corpus_sha256=reference["corpus_sha256"],
        )
    write_new(output / "evaluation-plan.json", settings)
    rows, losses = [], {}
    started = time.monotonic()
    reason = training_gate(recipe, started, yield_file)
    if not reason and recipe.device == "cuda":
        admission = gpu_gate(recipe.minimum_free_gpu_mib)
        if not admission["gpu_admitted"]:
            reason = admission["reason"]
    model, load_error = None, None
    if not reason:
        try:
            low_priority()
            torch.set_num_threads(recipe.cpu_threads)
            if recipe.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = PeftModel.from_pretrained(
                load_base(base, recipe),
                run / "best-adapter",
                is_trainable=False,
                local_files_only=True,
            )
            if reference:
                model.load_adapter(
                    reference_run / "best-adapter",
                    adapter_name="reference",
                    is_trainable=False,
                    local_files_only=True,
                )
        except (ValueError, RuntimeError, TypeError, OSError) as exc:
            load_error = f"{type(exc).__name__}: {exc}"
    if model is not None:
        model.eval()
    try:
        for arm in settings["arms"]:
            arm_error = load_error
            arm_pause = None
            if model and not arm_error:
                try:
                    model.set_adapter("reference" if arm == "reference" else "default")
                    model.eval()
                except (ValueError, RuntimeError, KeyError) as exc:
                    arm_error = f"Adapter selection failed: {type(exc).__name__}: {exc}"
            context = model.disable_adapter() if arm == "baseline" and model else nullcontext()
            with context:
                if model and not arm_error:
                    try:
                        losses[arm] = loss_on(
                            model,
                            encoded[split],
                            tokenizer,
                            recipe,
                            stop=lambda: training_gate(recipe, started, yield_file),
                        )
                    except YieldRequested as exc:
                        arm_pause = str(exc)
                    except (ValueError, RuntimeError, TypeError) as exc:
                        arm_error = f"Loss evaluation failed: {type(exc).__name__}: {exc}"
                for example in files[split]:
                    for seed in seeds:
                        row = {
                            "id": example["id"],
                            "arm": arm,
                            "seed": seed,
                            "status": "resource_wait",
                            "text": None,
                            "answer": None,
                            "wall_seconds": None,
                            "output_tokens": None,
                        }
                        reason = training_gate(recipe, started, yield_file)
                        if arm_error:
                            row.update(status="error", reason=arm_error)
                        elif reason or arm_pause or model is None:
                            row["reason"] = reason or arm_pause or "GPU admission was unavailable"
                        else:
                            call_start = time.monotonic()
                            stopped = []

                            class Stop(StoppingCriteria):
                                def __init__(self, call_start, stopped):
                                    self.call_start = call_start
                                    self.stopped = stopped

                                def __call__(self, input_ids, scores, **kwargs):
                                    pending = training_gate(recipe, started, yield_file)
                                    if (
                                        pending
                                        or time.monotonic() - self.call_start >= timeout_seconds
                                    ):
                                        self.stopped.append(pending or "Inference timeout")
                                        return True
                                    return False

                            try:
                                torch.manual_seed(seed)
                                prompt = tokenizer.apply_chat_template(
                                    example["prompt"],
                                    tokenize=True,
                                    add_generation_prompt=True,
                                    enable_thinking=recipe.enable_thinking,
                                )
                                prompt = token_ids(prompt)
                                row.update(
                                    input_tokens=len(prompt),
                                    prompt_tokens_sha256=digest(prompt),
                                )
                                if len(prompt) + max_new_tokens > recipe.max_length:
                                    raise ValueError(
                                        "Prompt plus generation exceeds frozen context budget"
                                    )
                                tokens = torch.tensor([prompt], device=recipe.device)
                                with torch.no_grad(), _context(recipe):
                                    generated = model.generate(
                                        input_ids=tokens,
                                        attention_mask=torch.ones_like(tokens),
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        stopping_criteria=StoppingCriteriaList(
                                            [Stop(call_start, stopped)]
                                        ),
                                    )[0, len(prompt) :]
                                text = tokenizer.decode(generated, skip_special_tokens=True)
                                terminal = int(generated[-1]) if len(generated) else None
                                stop_reason = (
                                    stopped[0]
                                    if stopped
                                    else "eos"
                                    if terminal == tokenizer.eos_token_id
                                    else "output_cap"
                                    if len(generated) >= max_new_tokens
                                    else "unclassified"
                                )
                                row.update(
                                    text=text,
                                    output_tokens=len(generated),
                                    terminal_token_id=terminal,
                                    stop_reason=stop_reason,
                                    output_cap_reached=len(generated) >= max_new_tokens,
                                    status=(
                                        "timeout"
                                        if stopped and stopped[0] == "Inference timeout"
                                        else "resource_wait"
                                        if stopped
                                        else "complete"
                                    ),
                                )
                                if stopped:
                                    row["reason"] = stopped[0]
                                else:
                                    expected = json.loads(example["completion"][0]["content"])
                                    try:
                                        packet = json.loads(example["prompt"][-1]["content"])
                                    except ValueError:
                                        packet = None
                                    row.update(diagnose(text, expected, packet))
                                    if row["json_valid"]:
                                        row["answer"] = json.loads(text)
                            except (ValueError, RuntimeError, TypeError) as exc:
                                row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                            row["wall_seconds"] = time.monotonic() - call_start
                        rows.append(row)
                        row["rss_bytes"] = psutil.Process().memory_info().rss
                        if model is not None and recipe.device == "cuda":
                            row["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
                            row["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
                        with (output / "answers.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(canonical(row) + "\n")
    finally:
        if model and recipe.quantization == "none":
            model.to("cpu")
        if recipe.device == "cuda":
            settings["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            settings["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            model = tokens = generated = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    summary = {
        **settings,
        "summary": summarize(rows),
        "completion_losses": losses,
        "status": "diagnostics_complete",
        "wall_seconds": time.monotonic() - started,
        "production_qualified": False,
        "comparison_usable": all(row["status"] == "complete" for row in rows),
        "economic_improvement": None,
        "test_consumed": split == "test",
        "rows_sha256": digest(rows),
        "limitations": [
            "Greedy repetitions/seeds are not independent evidence.",
            "Mechanical matching and loss are not semantic qualification.",
            "No activation or economic authority follows from this report.",
        ],
    }
    for loss in losses.values():
        if not math.isfinite(loss["loss"]):
            raise ValueError("Nonfinite comparison loss")
    write_new(output / "comparison.json", summary)
    return summary
