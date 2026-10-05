import copy

import pytest
from llm_lab.corpus import verify_corpus, write_corpus
from llm_lab.evaluation import diagnose, evaluate, summarize
from llm_lab.io import canonical, digest, read_object, sha_file
from llm_lab.masking import collate, encode, preflight, token_ids
from llm_lab.model import attach_adapter, load_base, tokenizer_at, verify_model
from llm_lab.training import train


def test_exact_target_and_padding_masking_with_real_template(hybrid):
    root, recipe = hybrid
    tokenizer = tokenizer_at(root / "base")
    _, files = verify_corpus(root / "corpus")
    receipt, encoded = preflight(files, tokenizer, recipe.max_length)
    rows = [encoded["train"][0], encoded["train"][1]]
    for row in rows:
        assert row["labels"][: row["prompt_tokens"]] == [-100] * row["prompt_tokens"]
        assert tokenizer.decode(row["labels"][row["prompt_tokens"] :], skip_special_tokens=True)
    batch = collate(rows, tokenizer, "cpu")
    assert ((batch["attention_mask"] == 0) & (batch["labels"] != -100)).sum() == 0
    assert receipt["splits"]["test"]["examples"] == 4


def test_overlong_packet_refused_instead_of_silently_truncated(hybrid):
    root, _ = hybrid
    _, files = verify_corpus(root / "corpus")
    with pytest.raises(ValueError, match="no silent truncation"):
        encode(files["train"][0], tokenizer_at(root / "base"), 16)


def test_nonprefix_chat_template_refused():
    class Renderer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return [1, 2] if add_generation_prompt else [1, 3, 4]

    with pytest.raises(ValueError, match="not prefix-preserving"):
        encode({"id": "test", "prompt": [], "completion": []}, Renderer(), 100)


def test_transformers_batchencoding_normalization():
    from transformers import BatchEncoding

    assert token_ids(BatchEncoding({"input_ids": [[1, 2, 3]]})) == [1, 2, 3]
    with pytest.raises(ValueError, match="exactly one"):
        token_ids({"input_ids": [[1], [2]]})


def test_optimizer_updates_only_adapters_and_retains_base_parameters(hybrid):
    import torch
    from llm_lab.training import loss_on

    root, recipe = hybrid
    torch.set_num_threads(2)
    tokenizer = tokenizer_at(root / "base")
    _, files = verify_corpus(root / "corpus")
    _, encoded = preflight(files, tokenizer, recipe.max_length)
    model, info = attach_adapter(load_base(root / "base", recipe), recipe)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    with model.disable_adapter():
        baseline_before = loss_on(model, encoded["validation"], tokenizer, recipe)["loss"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        model(**collate(encoded["train"][:2], tokenizer, "cpu")).loss.backward()
        optimizer.step()
    for name, original in before.items():
        assert torch.equal(dict(model.named_parameters())[name], original)
    with model.disable_adapter():
        assert loss_on(model, encoded["validation"], tokenizer, recipe)["loss"] == baseline_before
    assert info["trainable_parameters"] > 0


def test_pretrained_loading_receipt_is_serializable_and_complete(hybrid):
    root, recipe = hybrid
    model = load_base(root / "base", recipe)
    receipt = model._lab_loading_info
    assert receipt["text_weights_complete"] and receipt["missing_keys"] == []
    assert canonical(receipt)


def test_clean_yield_resume_matches_uninterrupted_optimizer_state(hybrid, tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file

    root, recipe = hybrid
    recipe = recipe.model_copy(update={"dropout": 0.1})
    original = verify_model(root / "base")
    uninterrupted = train(root / "base", root / "corpus", recipe, tmp_path / "whole")

    def yield_after_validation(*_):
        receipt_file = tmp_path / "yielded" / "run.json"
        completed = read_object(receipt_file)["step"] if receipt_file.exists() else 0
        return "Test requested a clean yield" if completed >= 2 else None

    with monkeypatch.context() as patch:
        patch.setattr("llm_lab.training.training_gate", yield_after_validation)
        yielded = train(root / "base", root / "corpus", recipe, tmp_path / "yielded")
    assert yielded["status"] == "yielded" and yielded["step"] == 2
    assert read_object(tmp_path / "yielded" / "checkpoint.json")["resume_safe"]
    resumed = train(
        root / "base", root / "corpus", recipe, tmp_path / "resumed", resume=tmp_path / "yielded"
    )
    assert resumed["status"] == "trained" and resumed["step"] == 6
    assert resumed["training_exposures"] == uninterrupted["training_exposures"]
    presentations = recipe.steps * recipe.micro_batch * recipe.accumulation
    assert sum(len(event["example_ids"]) for event in resumed["history"]) == presentations
    _, files = verify_corpus(root / "corpus")
    assert resumed["history"][-1]["effective_epochs"] == presentations / len(files["train"])
    assert verify_model(root / "base") == original
    assert uninterrupted["best_validation_loss"] == resumed["best_validation_loss"]
    first = load_file(tmp_path / "whole" / "last-adapter" / "adapter_model.safetensors")
    second = load_file(tmp_path / "resumed" / "last-adapter" / "adapter_model.safetensors")
    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_training_failure_is_retained_and_not_resumable(hybrid, tmp_path, monkeypatch):
    root, recipe = hybrid

    def fail(*_):
        raise RuntimeError("Deliberate model load failure")

    with monkeypatch.context() as patch:
        patch.setattr("llm_lab.training.load_base", fail)
        with pytest.raises(RuntimeError, match="Deliberate"):
            train(root / "base", root / "corpus", recipe, tmp_path / "failed")
    failed = read_object(tmp_path / "failed" / "run.json")
    assert failed["status"] == "failed"
    with pytest.raises(ValueError, match="yielded"):
        train(
            root / "base",
            root / "corpus",
            recipe,
            tmp_path / "bad-resume",
            resume=tmp_path / "failed",
        )


def test_yield_during_validation_resumes_that_validation_before_training(
    hybrid, tmp_path, monkeypatch
):
    from llm_lab.training import YieldRequested, loss_on

    root, recipe = hybrid
    uninterrupted = train(root / "base", root / "corpus", recipe, tmp_path / "whole")
    calls = 0

    def pause_first_adapter_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise YieldRequested("Protected hours began during validation")
        return loss_on(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr("llm_lab.training.loss_on", pause_first_adapter_validation)
        yielded = train(root / "base", root / "corpus", recipe, tmp_path / "paused")
    assert yielded["status"] == "yielded" and yielded["pending_validation_step"] == 2
    resumed = train(
        root / "base", root / "corpus", recipe, tmp_path / "resumed", resume=tmp_path / "paused"
    )
    assert resumed["pending_validation_step"] is None
    assert resumed["best_validation_loss"] == uninterrupted["best_validation_loss"]


def test_gis_priority_defers_before_model_load_or_artifact_creation(tmp_path, monkeypatch):
    from llm_lab.config import Recipe

    monkeypatch.setattr(
        "llm_lab.training.gpu_gate",
        lambda *_: {
            "gpu_admitted": False,
            "reason": "GIS priority hours",
        },
    )
    monkeypatch.setattr("llm_lab.training.load_base", lambda *_: pytest.fail("Model load"))
    result = train(
        tmp_path / "absent-base",
        tmp_path / "absent-corpus",
        Recipe(name="gpu", device="cuda", precision="float16"),
        tmp_path / "run",
    )
    assert result["status"] == "deferred"
    assert not (tmp_path / "run").exists()


def test_pair_model_load_failure_counts_every_requested_case(hybrid, tmp_path, monkeypatch):
    root, recipe = hybrid
    train(root / "base", root / "corpus", recipe, tmp_path / "run")

    def fail(*_):
        raise RuntimeError("Deliberate paired model load failure")

    monkeypatch.setattr("llm_lab.evaluation.load_base", fail)
    result = evaluate(tmp_path / "run", tmp_path / "comparison")
    for arm in ("baseline", "candidate"):
        assert result["summary"][arm]["requested"] == result["summary"][arm]["errors"] == 4
    assert result["comparison_usable"] is False


def test_pair_receipts_prove_same_prompt_and_explicit_stop_budget(hybrid, tmp_path):
    import json

    root, recipe = hybrid
    train(root / "base", root / "corpus", recipe, tmp_path / "run")
    result = evaluate(tmp_path / "run", tmp_path / "comparison", max_new_tokens=4)
    rows = [
        json.loads(line)
        for line in (tmp_path / "comparison/answers.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 8 and result["retries_per_case"] == 0
    for before, after in zip(rows[:4], rows[4:], strict=True):
        assert before["id"] == after["id"]
        assert before["prompt_tokens_sha256"] == after["prompt_tokens_sha256"]
        assert before["input_tokens"] == after["input_tokens"]
    for row in rows:
        assert row["output_tokens"] <= 4 and row["rss_bytes"] > 0
        assert row["stop_reason"] in {"eos", "output_cap"}
        assert (row["stop_reason"] == "eos") == (
            row["terminal_token_id"] in result["stop_token_ids"]
        )


def test_resume_refuses_changed_checkpoint(hybrid, tmp_path):
    root, recipe = hybrid
    paused = train(root / "base", root / "corpus", recipe, tmp_path / "paused", yield_after_step=1)
    assert paused["status"] == "yielded" and paused["step"] == 1
    with (tmp_path / "paused" / "optimizer.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint changed"):
        train(
            root / "base", root / "corpus", recipe, tmp_path / "refused", resume=tmp_path / "paused"
        )


def test_frozen_reference_adds_one_arm_with_identical_inputs_and_no_extra_attempts(
    hybrid, tmp_path
):
    import json

    root, recipe = hybrid
    reference = train(root / "base", root / "corpus", recipe, tmp_path / "reference")
    train(root / "base", root / "corpus", recipe, tmp_path / "run")
    result = evaluate(
        tmp_path / "run",
        tmp_path / "three-arms",
        reference_run=tmp_path / "reference",
        max_new_tokens=4,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "three-arms/answers.jsonl").read_text().splitlines()
    ]
    assert result["arms"] == ["baseline", "reference", "candidate"]
    assert result["reference_profile_sha256"] == reference["profile_sha256"]
    assert result["reference_candidate_files"] == reference["candidate_files"]
    assert result["retries_per_case"] == 0 and len(rows) == 12
    for arm in result["arms"]:
        assert result["summary"][arm]["requested"] == 4
    for group in zip(rows[:4], rows[4:8], rows[8:], strict=True):
        assert len({row["id"] for row in group}) == 1
        assert len({row["prompt_tokens_sha256"] for row in group}) == 1
    # Separate adapters with identical training must generate the same tiny-fixture output.
    for old, new in zip(rows[4:8], rows[8:], strict=True):
        assert old["text"] == new["text"] and old["output_tokens"] == new["output_tokens"]


def test_reference_tampering_is_refused_before_loading_or_consuming_cases(hybrid, tmp_path):
    root, recipe = hybrid
    train(root / "base", root / "corpus", recipe, tmp_path / "reference")
    train(root / "base", root / "corpus", recipe, tmp_path / "run")
    with (tmp_path / "reference/best-adapter/adapter_model.safetensors").open("ab") as stream:
        stream.write(b"tampered reference")
    with pytest.raises(ValueError, match="Frozen reference candidate changed"):
        evaluate(tmp_path / "run", tmp_path / "refused", reference_run=tmp_path / "reference")
    assert not (tmp_path / "refused").exists()


def test_reference_load_failure_records_every_requested_arm(hybrid, tmp_path, monkeypatch):
    from peft import PeftModel

    root, recipe = hybrid
    train(root / "base", root / "corpus", recipe, tmp_path / "reference")
    train(root / "base", root / "corpus", recipe, tmp_path / "run")
    original = PeftModel.load_adapter

    def load(self, *args, **kwargs):
        if kwargs.get("adapter_name") == "reference":
            raise RuntimeError("Deliberate frozen-reference load failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PeftModel, "load_adapter", load)
    result = evaluate(
        tmp_path / "run", tmp_path / "comparison", reference_run=tmp_path / "reference"
    )
    for arm in ("baseline", "reference", "candidate"):
        assert result["summary"][arm]["requested"] == result["summary"][arm]["errors"] == 4
    assert result["comparison_usable"] is False


def test_test_split_locked_and_failed_evaluation_calls_counted(hybrid, tmp_path):
    root, recipe = hybrid
    train(root / "base", root / "corpus", recipe, tmp_path / "run")
    with pytest.raises(ValueError, match="unlock-test"):
        evaluate(tmp_path / "run", tmp_path / "test", split="test")
    signal = tmp_path / "yield"
    signal.touch()
    result = evaluate(tmp_path / "run", tmp_path / "comparison", yield_file=signal)
    for arm in ("baseline", "candidate"):
        assert result["summary"][arm]["requested"] == 4
        assert result["summary"][arm]["resource_waits"] == 4
        assert result["summary"][arm]["complete"] == 0
    assert result["production_qualified"] is False
    assert result["economic_improvement"] is None


def test_selected_candidate_tampering_refused_before_evaluation(hybrid, tmp_path):
    root, recipe = hybrid
    receipt = train(root / "base", root / "corpus", recipe, tmp_path / "run")
    adapter = tmp_path / "run" / "best-adapter" / "adapter_model.safetensors"
    assert sha_file(adapter) == receipt["candidate_files"][adapter.name]
    with adapter.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="candidate changed"):
        evaluate(tmp_path / "run", tmp_path / "comparison")


def test_amended_reserved_cases_cannot_change_training_or_validation(hybrid, tmp_path):
    root, recipe = hybrid
    trained = train(root / "base", root / "corpus", recipe, tmp_path / "run")
    _, files = verify_corpus(root / "corpus")
    files = copy.deepcopy(files)
    files["test"][0]["prompt"][-1]["content"] += " Independent synthetic evaluation variant."
    files["test"][0]["id"] = digest(files["test"][0])
    amended = write_corpus(files, tmp_path / "amended", {"fixture": "synthetic review correction"})
    signal = tmp_path / "yield"
    signal.touch()
    result = evaluate(
        tmp_path / "run",
        tmp_path / "comparison",
        split="test",
        unlock_test=True,
        evaluation_corpus=tmp_path / "amended",
        yield_file=signal,
    )
    assert result["corpus_sha256"] == amended["corpus_sha256"]
    assert result["training_corpus_sha256"] == trained["corpus_sha256"]
    assert result["summary"]["candidate"]["requested"] == 4
    files["train"][0]["prompt"][-1]["content"] += " Changed training input."
    write_corpus(files, tmp_path / "changed-train", {"fixture": "synthetic prohibited change"})
    with pytest.raises(ValueError, match="cannot change training or validation"):
        evaluate(
            tmp_path / "run",
            tmp_path / "refused",
            split="test",
            unlock_test=True,
            evaluation_corpus=tmp_path / "changed-train",
        )


def test_mechanical_critique_counts_unsafe_approval_missing_evidence_and_false_rejection():
    expected = {"action": "reject", "evidence_ids": ["e0"]}
    assert diagnose('{"action":"approve"}', expected)["critical"]
    assert diagnose(
        '{"action":"reject","evidence_ids":["invented"]}', expected, {"evidence": {"e0": {}}}
    )["critical"]
    assert diagnose('{"action":"reject"}', {"action": "propose_experiment"})["false_rejection"]
    assert not diagnose("unparseable", expected)["json_valid"]
    rows = [
        {"arm": "candidate", "status": status}
        for status in ("complete", "timeout", "error", "resource_wait")
    ]
    summary = summarize(copy.deepcopy(rows))["candidate"]
    assert summary["requested"] == 4 and summary["complete"] == 1
    assert summary["timeouts"] == summary["errors"] == summary["resource_waits"] == 1


def test_amp_overflow_skips_weight_update_and_retains_bounded_retry_receipt():
    import torch
    from llm_lab.config import Recipe
    from llm_lab.training import clip_or_backoff

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    recipe = Recipe(name="overflow-test", max_amp_overflow_retries=1)
    receipt = {"step": 0}
    for attempt in range(2):
        optimizer.zero_grad()
        scaler.scale(parameter.sum() * float("inf")).backward()
        scaler.unscale_(optimizer)
        if attempt == 0:
            assert clip_or_backoff([parameter], optimizer, scaler, recipe, receipt) is None
            assert parameter.item() == 1.0 and scaler.get_scale() == 4
        else:
            with pytest.raises(RuntimeError, match="retry limit"):
                clip_or_backoff([parameter], optimizer, scaler, recipe, receipt)
            assert parameter.item() == 1.0
    assert len(receipt["amp_overflow_retries"]) == 2 and receipt["step"] == 0


def test_nonfinite_unscaled_gradients_are_fatal():
    import torch
    from llm_lab.config import Recipe
    from llm_lab.training import clip_or_backoff

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="Nonfinite gradients"):
        clip_or_backoff(
            [parameter],
            optimizer,
            torch.amp.GradScaler("cpu", enabled=False),
            Recipe(name="bad-gradient"),
            {"step": 0},
        )
    assert parameter.item() == 1.0
