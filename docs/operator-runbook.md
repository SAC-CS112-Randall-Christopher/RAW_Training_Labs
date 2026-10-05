# Operator runbook

## Before a model job

Open the intended project and verify its identifier, task, declared criteria and frozen corpus. Inspect review type/authorship, provenance and permissions instead of inferring customer consent. A changed correction or permission requires an explicit review revision before freeze. A frozen population cannot be edited in place; use a fresh independently reviewed population when further work is appropriate.

The initial demonstration is restricted to RAW-authored synthetic material, a pinned Qwen3-0.6B public checkpoint, CPU LoRA, two CPU threads, eight GiB owned-worker RSS, six successful steps, 512 training-context tokens and a 1,800-second training watchdog. Other initial model calls have explicit token/time settings; acquisition has a 600-second watchdog. The private data store has a 20 GiB ceiling and the separate C: runtime has a five GiB allowance. The original Lab and Q-Trades remain separate.

Baseline, candidate, recovery, final comparison and export must match exact retained identities. If an identity changes, explicitly evaluate a new configured baseline instead of overriding the check. Train/validation label handling and sealed-final behavior can be inspected without opening original private holdouts.

## Read job states correctly

| State | Meaning and next action |
| --- | --- |
| starting / running | A durable owned worker was admitted. Use its job identifier and inspect recorded progress/logs. Another heavy job is blocked across projects. |
| cancelling | A cooperative stop was requested. Wait for a terminal state; do not infer a checkpoint. |
| completed | The requested stage finished. For a train job, task acceptance still depends on the matched final comparison. For an export, runtime verification and task quality remain separate. |
| yielded | Training saved a clean checkpoint. Check the retained receipt and choose Resume clean checkpoint, which creates a new attempt linked to the prior one. |
| cancelled | Inspect checkpoint availability explicitly. A completed acquisition is not implied; partial public-model files can be retained for another deliberate acquisition. |
| interrupted | The owned worker disappeared before a complete result. Its partial files/logs remain evidence. A safe optimizer checkpoint is not implied. Start a new candidate or acquisition deliberately; never label it a successful resume. |
| deferred / failed | Read the retained reason. Correct the specified resource, input, compatibility or execution issue before another explicit attempt. No hidden retries promote failure to success. |

Closing and reopening the browser does not cancel a job. Terminating only the RAW web-server process also permits an independent worker to continue. A development terminal's process-tree cancellation can stop both server and worker; the workbench retains that interruption honestly. Restart the server, select the same project, and inspect the same job from either GUI or CLI.

## Pause, resume and cancel

Use **Save checkpoint & pause** for an active train job, or the corresponding CLI `yield`. The trainer reaches a safe boundary, saves optimizer/scaler/RNG and adapter identities, and reports yielded when complete. The bounded demo's step-three pause is an explicit successful-step boundary, not a fabricated checkpoint.

Resume only the same owned project/profile/corpus and baseline. The resumed attempt consumes the previous clean checkpoint and continues successful-step accounting; it does not replay an unrelated model/data recipe. A changed checkpoint, source or environment blocks recovery. Abruptly killed or failed jobs are not automatically resumable.

Cancel writes a cooperative request for ordinary stages. Training turns the request into a safe checkpoint pause where possible; inspect its actual terminal receipt. Acquisition cancellation forces an owned stop and retains partial transfer files. Resource/watchdog failures stop owned runtime children and the worker even if the store cannot persist a failure report; later reconciliation reports interruption instead of success.

## Compare and deliver

Use the candidate's recorded baseline and frozen profile. The final comparison evaluates baseline and candidate on identical held-out/regression inputs, settings and seeds. Error and malformed outputs count as failures. Exact matching is the declared synthetic extraction measure; unsupported-field counts are a specific proxy, not a general hallucination detector. Record substantive reviewer disagreement separately.

If criteria fail or quality worsens, keep the result in development and preserve the evidence. A package may still be exported for explicitly authorized development/runtime verification, with its failed criteria and limitations visible. Export does not install, publish, activate or prove customer acceptance.

The PEFT package needs its exact original base model; do not substitute an Ollama GGUF file. `verify_runtime.py` can run with a separate Python process and the package's exact base directory. Inspect its receipt and output match. Altered package/base hashes are refused. Its source license and RAW export/retention rights must accompany the handoff; model artifacts may retain source information.

## Capability gates

Tev1 4B is RAW's preferred System 1 decision profile, and Tev1 0.8B is the experimental comparison. Installed metadata verifies their GGUF identities; it does not prove compatible trainable source, release rights, local resource approval, or RAW performance. The decision template uses state/question/labeled options with a single-letter answer. BexDog's measured preference is user-supplied context, not a RAW benchmark.

Qwen3.5 4B reasoning coverage currently requires a separate approved resource envelope and GPU environment. Inherited tiny hybrid checks are software evidence only. Qwen3 8B is recorded as BexDog's heavier fallback. Preserve GPU priority for GIS during the approved weekday working hours. GRPO and other RL methods remain a later milestone.

## Routine validation

Use the shared CLI to verify GUI project/corpus/job identifiers and result locations. Use malformed and wrong-project inputs to confirm truthful errors before private-customer work. Preserve original inputs, review history and unsuccessful attempts. Never move operational files into the source tree to make a test convenient.

The source-only import receipt identifies the original committed engine. Verify the original Lab's checkout, original Q-Trades state and installed configuration have not changed as part of this lane. Only source and nonsensitive proof summaries belong in the private repository; dataset, weight and package directories remain in the separate owned store.
