"""Bounded token-normalized SFT, validation selection, durable yield/resume receipts."""

import gc
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

from llm_lab.config import Recipe
from llm_lab.corpus import verify_corpus
from llm_lab.io import digest, new_directory, read_object, sha_file, write_new, write_state
from llm_lab.masking import collate, preflight
from llm_lab.model import (
    attach_adapter,
    environment,
    load_base,
    source_identity,
    tokenizer_at,
    verify_model,
)
from llm_lab.resources import gpu_gate, low_priority


class YieldRequested(RuntimeError):
    """A cooperative resource wait, never a model-quality failure."""


def _context(recipe):
    import torch

    return (
        torch.autocast("cuda", dtype=getattr(torch, recipe.precision))
        if recipe.device == "cuda" and recipe.precision != "float32"
        else nullcontext()
    )


def supervised_count(batch):
    return int((batch["labels"][:, 1:] != -100).sum().item())


def clip_or_backoff(parameters, optimizer, scaler, recipe, receipt):
    import torch

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all())
    if not finite:
        if not scaler.is_enabled():
            raise RuntimeError("Nonfinite gradients without mixed-precision scaling")
        previous_scale = scaler.get_scale()
        # unscale_ already recorded the inf/NaN. This step skips optimizer mutation;
        # update lowers the scale. Never clip nonfinite gradients or count a skipped step.
        scaler.step(optimizer)
        scaler.update()
        event = {
            "optimizer_step": receipt["step"] + 1,
            "previous_scale": previous_scale,
            "new_scale": scaler.get_scale(),
            "gradient_finite": False,
        }
        retries = receipt.setdefault("amp_overflow_retries", [])
        retries.append(event)
        count = sum(item["optimizer_step"] == event["optimizer_step"] for item in retries)
        if count > recipe.max_amp_overflow_retries:
            raise RuntimeError("Declared mixed-precision overflow retry limit exceeded")
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm, error_if_nonfinite=True)


def loss_on(model, values, tokenizer, recipe, *, stop=None):
    import torch

    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(values), recipe.micro_batch):
            reason = stop() if stop else None
            if reason:
                raise YieldRequested(reason)
            batch = collate(values[start : start + recipe.micro_batch], tokenizer, recipe.device)
            count = supervised_count(batch)
            with _context(recipe):
                loss = float(model(**batch).loss.item())
            if not math.isfinite(loss) or count <= 0:
                raise ValueError("Nonfinite evaluation or absent target tokens")
            loss_sum += loss * count
            tokens += count
    return {"loss": loss_sum / tokens, "supervised_tokens": tokens, "examples": len(values)}


def training_gate(recipe, started, yield_file=None):
    if yield_file and Path(yield_file).exists():
        return "Explicit training yield requested"
    if time.monotonic() - started >= recipe.max_run_seconds:
        return "Declared run time budget reached"
    if recipe.device == "cuda":
        state = gpu_gate(recipe.minimum_free_gpu_mib, check_memory=False)
        if not state["gpu_admitted"]:
            return state["reason"]
    return None


def train(
    base: Path,
    corpus: Path,
    recipe: Recipe,
    destination: Path,
    *,
    resume: Path | None = None,
    yield_file: Path | None = None,
    yield_after_step: int | None = None,
) -> dict:
    import torch

    started = time.monotonic()
    low_priority()
    torch.set_num_threads(recipe.cpu_threads)
    if recipe.device == "cuda":
        gate = gpu_gate(recipe.minimum_free_gpu_mib)
        if not gate["gpu_admitted"]:
            return {"status": "deferred", "reason": gate["reason"], "resource_state": gate}
    base_manifest = verify_model(base)
    corpus_manifest, files = verify_corpus(corpus, splits=("train", "validation"))
    if not files.get("train") or not files.get("validation"):
        raise ValueError("Training requires reviewed training and independent validation inputs")
    tokenizer = tokenizer_at(base)
    masking, encoded = preflight(
        files, tokenizer, recipe.max_length, enable_thinking=recipe.enable_thinking
    )
    identity = {
        "base_sha256": base_manifest["base_sha256"],
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "recipe": recipe.model_dump(),
        "masking": masking,
        "environment": environment(),
        "source": source_identity(),
    }
    profile_hash = digest(identity)
    previous = None
    if resume:
        previous = read_object(resume / "run.json")
        if previous["profile_sha256"] != profile_hash or previous["status"] != "yielded":
            raise ValueError(
                "Resume requires a yielded run with identical base/data/recipe/source/environment"
            )
        if not (resume / "checkpoint.json").is_file():
            raise ValueError("No complete optimizer checkpoint available")
    run = new_directory(destination)
    receipt = {
        **identity,
        "profile_sha256": profile_hash,
        "status": "running",
        "step": 0,
        "best_validation_loss": None,
        "best_step": None,
        "history": [],
        "training_exposures": {
            "attempted_presentations": 0,
            "attempted_input_tokens": 0,
            "attempted_supervised_tokens": 0,
        },
        "resumed_from": str(resume.resolve()) if resume else None,
        "base_directory": str(base.resolve()),
        "corpus_directory": str(corpus.resolve()),
        "limitations": [
            "Loss and mechanical scores are not semantic or product qualification.",
            "GIS yielding is cooperative at step boundaries, not hardware preemption.",
        ],
    }
    write_state(run / "run.json", receipt)
    model, optimizer, error = None, None, None
    try:
        torch.manual_seed(recipe.seed)
        if recipe.device == "cuda":
            torch.cuda.manual_seed_all(recipe.seed)
            torch.cuda.reset_peak_memory_stats()
        model = load_base(base, recipe)
        receipt["pretrained_loading"] = getattr(model, "_lab_loading_info", {})

        def pause():
            return training_gate(recipe, started, yield_file)

        baseline = loss_on(
            model,
            encoded["validation"],
            tokenizer,
            recipe,
            stop=pause,
        )
        model, adapter_info = attach_adapter(model, recipe)
        receipt.update(adapter_info)
        receipt["baseline_validation"] = baseline
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=recipe.device == "cuda" and recipe.precision == "float16"
        )
        step = 0
        if previous:
            checkpoint = read_object(resume / "checkpoint.json")
            if checkpoint.get("resume_safe") is not True:
                raise ValueError("Checkpoint did not complete a clean yield")
            for name, value in checkpoint["files"].items():
                if Path(name).name != name or sha_file(resume / name) != value:
                    raise ValueError("Resume checkpoint changed")
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            for name, file_hash in checkpoint["adapter_files"].items():
                if Path(name).name != name or sha_file(resume / "last-adapter" / name) != file_hash:
                    raise ValueError("Resume adapter changed")
            set_peft_model_state_dict(
                model,
                load_file(resume / "last-adapter" / "adapter_model.safetensors"),
                adapter_name="default",
            )
            state = torch.load(
                resume / "optimizer.pt", map_location=recipe.device, weights_only=True
            )
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            torch.set_rng_state(state["cpu_rng"].cpu())
            if recipe.device == "cuda":
                # Loading optimizer tensors onto CUDA also relocates RNG byte tensors;
                # CUDA generators require their serialized states on the CPU.
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
            step = state["step"]
            if step != checkpoint["step"] or step != previous["step"]:
                raise ValueError("Resume receipt and optimizer steps disagree")
            receipt.update(
                step=step,
                history=list(previous["history"]),
                best_validation_loss=previous["best_validation_loss"],
                best_step=previous["best_step"],
                pending_validation_step=previous.get("pending_validation_step"),
                amp_overflow_retries=list(previous.get("amp_overflow_retries", [])),
                training_exposures=dict(previous.get("training_exposures", {})),
            )
            if (resume / "best-adapter").exists():
                import shutil

                for name, file_hash in checkpoint["best_adapter_files"].items():
                    if (
                        Path(name).name != name
                        or sha_file(resume / "best-adapter" / name) != file_hash
                    ):
                        raise ValueError("Resume selected adapter changed")
                shutil.copytree(resume / "best-adapter", run / "best-adapter")

        def record_validation(event):
            receipt["pending_validation_step"] = step
            validation = loss_on(model, encoded["validation"], tokenizer, recipe, stop=pause)
            event["validation_loss"] = validation["loss"]
            best = receipt["best_validation_loss"]
            if best is None or validation["loss"] < best:
                model.save_pretrained(
                    run / "best-adapter",
                    safe_serialization=True,
                    selected_adapters=[model.active_adapter],
                )
                receipt.update(best_validation_loss=validation["loss"], best_step=step)
            receipt["pending_validation_step"] = None

        if receipt.get("pending_validation_step") is not None:
            record_validation(receipt["history"][-1])
        values = encoded["train"]
        while step < recipe.steps:
            if yield_after_step is not None and step >= yield_after_step:
                receipt.update(
                    status="yielded", reason="Explicit successful-step checkpoint requested"
                )
                break
            reason = training_gate(recipe, started, yield_file)
            if reason:
                receipt.update(status="yielded", reason=reason)
                break
            step_started = time.monotonic()
            # Each optimizer step has deterministic shuffled sampling without test selection.
            order = list(range(len(values)))
            random.Random(recipe.seed + step // max(1, len(values))).shuffle(order)
            group = [
                values[order[(step * recipe.micro_batch * recipe.accumulation + i) % len(order)]]
                for i in range(recipe.micro_batch * recipe.accumulation)
            ]
            batches = [
                collate(group[i : i + recipe.micro_batch], tokenizer, recipe.device)
                for i in range(0, len(group), recipe.micro_batch)
            ]
            target_count = sum(supervised_count(batch) for batch in batches)
            exposures = receipt["training_exposures"]
            exposures["attempted_presentations"] = exposures.get(
                "attempted_presentations", 0
            ) + len(group)
            exposures["attempted_input_tokens"] = exposures.get("attempted_input_tokens", 0) + sum(
                int(b["attention_mask"].sum().item()) for b in batches
            )
            exposures["attempted_supervised_tokens"] = (
                exposures.get("attempted_supervised_tokens", 0) + target_count
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            total = 0.0
            for batch in batches:
                with _context(recipe):
                    loss = model(**batch).loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss; retained run cannot complete")
                weighted = loss * supervised_count(batch) / target_count
                total += float(weighted.detach())
                scaler.scale(weighted).backward()
            scaler.unscale_(optimizer)
            norm = clip_or_backoff(
                [p for p in model.parameters() if p.requires_grad],
                optimizer,
                scaler,
                recipe,
                receipt,
            )
            if norm is None:
                write_state(run / "run.json", receipt)
                continue
            if step < recipe.warmup_steps:
                rate = recipe.learning_rate * (step + 1) / max(1, recipe.warmup_steps)
            else:
                fraction = (step - recipe.warmup_steps) / max(1, recipe.steps - recipe.warmup_steps)
                rate = recipe.learning_rate * 0.5 * (1 + math.cos(math.pi * fraction))
            for param_group in optimizer.param_groups:
                param_group["lr"] = rate
            scaler.step(optimizer)
            scaler.update()
            step += 1
            receipt["step"] = step
            event = {
                "step": step,
                "training_loss": total,
                "learning_rate": rate,
                "target_tokens": target_count,
                "gradient_norm": float(norm),
                "elapsed_seconds": time.monotonic() - started,
                "step_wall_seconds": time.monotonic() - step_started,
                "input_tokens": sum(int(b["attention_mask"].sum().item()) for b in batches),
                "example_ids": [row["id"] for row in group],
                "successful_presentations": step * recipe.micro_batch * recipe.accumulation,
                "effective_epochs": step * recipe.micro_batch * recipe.accumulation / len(values),
            }
            receipt["history"].append(event)
            if step % recipe.eval_every == 0 or step == recipe.steps:
                record_validation(event)
            write_state(run / "run.json", receipt)
        if receipt["status"] == "running":
            receipt["status"] = "trained"
        if receipt["best_step"] is not None:
            receipt["checkpoint_selection"] = "Lowest validation completion loss; never test"
    except YieldRequested as exc:
        receipt.update(status="yielded" if optimizer is not None else "deferred", reason=str(exc))
    except BaseException as exc:
        receipt.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        error = exc
    finally:
        try:
            if model is not None and optimizer is not None:
                model.save_pretrained(
                    run / "last-adapter",
                    safe_serialization=True,
                    selected_adapters=[model.active_adapter],
                )
                state = {
                    "optimizer": optimizer.state_dict(),
                    "step": receipt["step"],
                    "scaler": scaler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if recipe.device == "cuda" else [],
                }
                torch.save(state, run / "optimizer.pt")
                write_new(
                    run / "checkpoint.json",
                    {
                        "files": {"optimizer.pt": sha_file(run / "optimizer.pt")},
                        "adapter_files": {
                            p.name: sha_file(p)
                            for p in (run / "last-adapter").iterdir()
                            if p.is_file()
                        },
                        "best_adapter_files": {
                            p.name: sha_file(p)
                            for p in (run / "best-adapter").glob("*")
                            if p.is_file()
                        },
                        "step": receipt["step"],
                        "resume_safe": receipt["status"] == "yielded",
                    },
                )
            if receipt["status"] == "trained":
                receipt["candidate_files"] = {
                    p.name: sha_file(p) for p in (run / "best-adapter").iterdir() if p.is_file()
                }
            verify_model(base)
        except BaseException as checkpoint_error:
            receipt.update(status="failed", checkpoint_error=str(checkpoint_error))
            error = error or checkpoint_error
        try:
            if model is not None and recipe.quantization == "none":
                model.to("cpu")
            if recipe.device == "cuda":
                receipt["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                receipt["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
                # Quantized bases cannot use .to('cpu'). Drop every local GPU
                # owner before empty_cache, including saved optimizer tensors.
                model = optimizer = scaler = state = batches = batch = loss = weighted = norm = None
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                receipt["gpu_allocated_after_release_bytes"] = torch.cuda.memory_allocated()
                receipt["gpu_reserved_after_release_bytes"] = torch.cuda.memory_reserved()
        except BaseException as release_error:
            receipt.update(status="failed", release_error=str(release_error))
            error = error or release_error
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["finished_at"] = time.time()
        write_state(run / "run.json", receipt)
    if error is not None:
        raise error
    return receipt
