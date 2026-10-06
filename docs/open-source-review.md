# Open-source training review

Requested by Chris Randall on 2026-10-05. Starting point: the supplied GitHub search for training labs, narrowed to LLM fine-tuning. The exact search page did not render through the research browser; GitHub's repository-search API and the projects' own documentation supplied the evidence. Repository names alone were not treated as proof of a usable training backend.

This is a source/documentation investigation. No third-party training framework was installed or benchmarked on RAW hardware during this investigation. Upstream performance and hardware claims are not RAW measurements. No customer examples, original Lab artifacts, BexDog state or private holdouts were shared with these projects.

## Candidates and decisions

| Project | Evidence and useful lesson | RAW decision |
| --- | --- | --- |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | Apache-2.0; CLI and LlamaBoard GUI, named model templates, SFT/LoRA/QLoRA, explicit dataset column mappings. Its README distinguishes thinking and non-thinking templates and requires identical templates at training and inference. | Strongest future GUI/backend reference. Preserve RAW's shared durable services and reviewed rights/split authority. Adopt its template discipline now; assess a pinned backend only in a separate compatibility milestone. |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Apache-2.0; YAML configurations and model-specific adapters. The official conversation guide exposes roles trained, EOS/EOT handling, tokenizer templates and custom message columns. | Useful for inspectable recipe design and token-level checks. Do not infer that its recommended GPU/attention settings fit RAW's T1000. A backend would need measured resource, resume, isolation and export checks. |
| [Hugging Face PEFT](https://github.com/huggingface/peft) | Apache-2.0; trains additional adapter parameters and documents adapter configuration/weights separately from their base model. | Reuse directly: the inherited RAW engine already uses PEFT. Keep exact base identity, immutable hashes and independent loading; an adapter alone does not establish portability or quality. |
| [Hugging Face TRL](https://github.com/huggingface/trl) | SFTTrainer supports PEFT and distinguishes completion-only from assistant-only loss. Assistant masks depend on generation markers in the template; known templates may be patched. | Reference for explicit label validation and future trainer compatibility. Keep the tested inherited completion mask for this bounded workflow. A trainer migration must preserve token weighting, checkpoint identity and matched evaluation. GRPO remains later. |
| [Unsloth](https://github.com/unslothai/unsloth) | Its README describes local desktop/Studio training. Core is Apache-2.0; optional Studio UI has AGPL-3.0 terms. Reported performance/support depends on the selected model, component and environment. | Candidate for a later measured acceleration/export experiment. Do not copy Studio UI into this private product or run its installation scripts without a component-specific licensing and compatibility decision. |
| [SwanLab](https://github.com/SwanHubX/SwanLab) | Apache-2.0 experiment tracking and visualization; repository describes cloud and self-hosted operation and several trainer integrations. | Learn from run-history and metric visualization. RAW's small first workflow already retains local SQLite records and immutable receipts; adding a tracking service is not currently needed. Any integration must keep source data and artifacts local. |
| [Post-Training Lab](https://github.com/uygarkurt/post-training-lab) | MIT, small readable SFT/GRPO/evaluation/inference scripts, separate MLX and CUDA paths. Its reported CUDA experiments use different hardware from RAW. | Good educational reference for simple, inspectable training loops. Its backend/hardware results do not validate RAW's CPU/T1000 envelope. Do not make GRPO a prerequisite. |
| [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) | The current README explicitly says it is no longer maintained and recommends Axolotl, TRL or Transformers Trainer. Its historical local/cloud task workflows illustrate column mapping and separate modalities. | Historical UX reference only; avoid it as a new foundational dependency. Non-generative classification and sentence-transformer support in another tool does not confer that support on RAW's generative engine. |

## Concrete lessons for the current workflow

1. **Reuse established primitives.** Transformers, PEFT and safetensors already provide model loading, LoRA parameters and adapter serialization. RAW supplies project ownership, review/permissions, sealed split handling, durable jobs and task acceptance.
2. **Make behavior part of identity.** Record tokenizer/template hashes, thinking mode, effective completion labels, context and decoding settings with every admitted model job. Fail incompatible comparisons and recovery explicitly. Never assume System 1/System 2 imply different architectures.
3. **Inspect the labels before training.** The same conversation can train different tokens under different collators/templates. Expose trainable versus ignored token counts, reject overlong examples, and test EOS/padding/prefix behavior on the actual tokenizer. A generic fallback template is insufficient evidence.
4. **Keep successful training separate from task acceptance.** Retain loss and validation checkpoint selection; compare matched baseline/candidate final cases for exact fields, invalid outputs, unsupported field claims, regressions, errors, latency and memory. Response length is diagnostic only.
5. **Treat export as another proof stage.** Retain exact base and adapter hashes, licenses, mode, recipe, environment and limitations; load the exported package in a separate runtime process. A GGUF download is not automatically a trainable PEFT checkpoint.
6. **Add backend breadth only with evidence.** The next backend checkpoint must cover a known compatible model, approved resources, identical data authority, one-job admission, bounded cancellation/resume, failures and independent runtime loading. QLoRA, non-generative classification/reranking and RL each need separate honest status.

These decisions are engineering inferences from the cited upstream documentation and RAW's requirements. No LLaMA-Factory, Axolotl, Unsloth or other application source was copied into RAW by this review. A future code import should pin its revision, preserve required attribution/license notices, and record its modified files and outcome checks.

## Authoritative details

- [LLaMA-Factory dataset formats and mapping](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)
- [Axolotl conversation templates, role masks and EOS/EOT](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)
- [TRL SFT labels, metrics and PEFT integration](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT checkpoint contents and base-model relationship](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [Unsloth component licensing](https://github.com/unslothai/unsloth#license)
- [Transformers chat-template behavior](https://huggingface.co/docs/transformers/main/chat_templating)
- [Axolotl's bounded debugging recommendations](https://docs.axolotl.ai/docs/debugging.html)

Recheck these moving upstream branches before an implementation dependency is selected. This review is a decision record, not a promise that every advertised method or model runs in RAW.
