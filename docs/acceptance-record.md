# Local acceptance record

Recorded on 2026-10-05 for the approved independent RAW Training Labs workflow. This is local software and synthetic-model evidence, not customer acceptance, a production rollout, or proof of all registered capability profiles.

## Scope and authority

The operator GUI and CLI use the same project-isolated services, reviewed originals, frozen manifests, durable job records and bounded owned workers. The only pretrained training demonstration used RAW-authored synthetic service notes and the approved Qwen/Qwen3-0.6B CPU LoRA profile. Review authorship was explicitly delegated Codex authorship; no independent human review or customer-data rights were inferred.

The final Python source identity is `4ca0cf6330b2278e09d3fe0f2933e37b238a8b833a30c6f18139489db49975ff`. The final baseline, training, resumed attempt, comparison and export must preserve this admitted source identity. The exact public base revision is `c1899de289a04d12100db370d81485cdf75e47ca`; its retained base manifest SHA-256 is `06513f21c553ee204e0ed98db4035a28ac5d750727d4756e0bde64083dd51ffe`.

Operational examples, receipts, checkpoints, base weights and packages are retained outside Git in `G:\Projects\RAW-Training-Labs-Data`. The independent CPU environment is `C:\Projects\RAW-Training-Labs-Runtime\cpu`; both RAW packages resolve to the managed implementation worktree, not the original Lab. No original environment or model directory was reused.

## Final source workflow

Project `3579ff0267d64b60901e47e924d35b57` contains 36 training, 8 validation, 8 sealed final-test and 8 regression examples. Frozen corpus `090224e6b3bb4504a18785695c13908e` has SHA-256 `53cb8654ad51b6469a9c3b7318a28a5070597a484d5cd877acd67c84d2593acb`. Actual tokenizer preflight checked all train/validation labels without truncation: maximum 152 encoded tokens, 1,128 supervised training tokens and 259 supervised validation tokens. Prompt and padding labels are ignored; the recorded non-thinking template and generation settings are preserved.

| Stage | Durable job | Observed outcome |
| --- | --- | --- |
| Base verification | `ee6e803587f64d6fb15907a42189b69d` | Completed; exact frozen public files were copied within the owned RAW store and verified. This was not another network download or reuse of the original Lab's model. |
| Configured baseline | `692bd5023ce24088a14f18476a64de1b` | Completed before training: 6/16 successful validation/regression cases, 7 invalid outputs and 0 model-call failures. These are not the final comparison's baseline scores. |
| Train and pause | `52d639d5dec74f9a90840cc58b36c133` | Yielded at 3/6 successful steps with a verified clean checkpoint; selected step 2 before recovery. |
| Resume | `8b095764538e465296d8fe634cd08f9a` | New attempt resumed the verified prior checkpoint and completed 6/6 successful steps. Selected step 6 by validation completion loss, never final-test score. |
| Final matched comparison | `3aa85afeb3c64ee59fb20d61a856ff5b` | Completed; actual owned-process wait observed exit code 0. Declared quality criteria failed; retain the selected candidate in development. |
| Development export/runtime | `6771516e7a144e83942f68443a6d722a` | Completed; standalone package load reproduced its retained candidate probe, with failed quality criteria preserved and no activation. |

Training used the unchanged float32 LoRA recipe, rank 16/alpha 32, 10,092,544 trainable parameters, two CPU threads, microbatch 1/accumulation 4 and 512 context tokens. Initial validation completion loss was 0.524742114; selected final validation loss was 0.000558200. Loss is not task acceptance. The pause and resumed attempts took approximately 128 and 135 seconds and peaked at 3.41 GiB owned-worker RSS, within the approved 8 GiB ceiling. Reopening the owned browser recovered the completed resumed attempt with the same identifier and retained 6/6 steps.

The final comparison used 8 sealed test and 8 regression cases, with identical input/profile/template/mode/generation/seed per pair. It took approximately 291 seconds and peaked at 3.38 GiB owned-worker RSS. The separate process observer retained its actual exit code in `checks\operator-evidence\final-comparison-process-exit-3aa85afeb3c64ee59fb20d61a856ff5b.json`.

| Final paired measure | Baseline | Candidate |
| --- | --- | --- |
| Exact whole-case success | 7/16 (43.75%) | 11/16 (68.75%) |
| Exact field accuracy | 57.8125% | 90.625% |
| Invalid outputs | 6 | 0 |
| Unsupported-field claim proxy | 3 | 6 |
| Model-call failures | 0 | 0 |
| p95 response time | 9.985 seconds | 8.204 seconds |
| Peak observed inference RSS | 2.69 GiB | 2.80 GiB |

There were four improved cases and zero whole-case success regressions. The candidate missed the declared minimum 80% whole-case success. Invalid outputs remain in the denominator. The unsupported-field count is an exact-target proxy, including field-content differences; it is not a general semantic hallucination detector, and invalid baseline JSON is not parsed for that proxy. The candidate still invents a site when none is stated. A delegated Codex case assessment was actually saved through the repaired GUI and read identically from the shared CLI; it does not change automatic scores or constitute independent human/customer acceptance. No later checkpoint or recipe was selected against these exposed final cases.

The exported package is retained under the final project's `jobs\6771516e7a144e83942f68443a6d722a\package`. Its package-manifest SHA-256 is `fa66c1cd49acb8a82ad647ba30438ab3014b050aa0fe7b70ce1d2daaeae8916c`. A standalone `python -I` process loaded the adapter against the exact verified base and reproduced the recorded candidate output. The receipt reports `status: verified`, `no_lab_imports: true`, `quality_acceptance: false` and `activated: false`; its probe took 7.031 seconds. The complete export/verification job took 42.094 seconds and peaked at 3.33 GiB owned RSS, including its runtime child. The verifier imports Transformers/PEFT, not RAW or the original Lab. It uses the separately installed CPU environment; this is not proof of an arbitrary new machine/environment, a GGUF conversion, customer acceptance or activation.

The actual delivery page now places the verified load and failed task criteria ahead of a collapsed technical receipt. A missing verification receipt is not presented as a verified package. This is a presentation-only change; the admitted Python/model source identity remains unchanged.

All six final-project job identity receipts were checked against the unchanged full Python source identity; every exported package file and the package-manifest hash were reverified. No heavy RAW job remained active. The final owned data store occupied 7.934 GiB of its 20 GiB allowance; the entire separate C: runtime, dependency caches and proof directory occupied 0.724 GiB of its 5 GiB allowance. These are actual stored bytes, not total-machine disk/RAM claims. Actual comparison and delivery screenshots are retained in this chat's local visualization directory, outside Git.

## Recovery and GUI/CLI evidence

The actual browser exercised project creation, neutral import, explicit delegated review, freeze, tokenizer preflight, authorization, baseline, training, pause, resume and comparison admission. Closing/reopening the owned browser recovered the same durable job. During the final-source comparison, verified worker PID 7040 remained running with the identical job/creation identity after browser closure and reopening; the local `checks\operator-evidence\final-source-browser-recovery-3aa85afeb3c64ee59fb20d61a856ff5b.json` records the observed before/after states. A prior owned-server-only termination also left a detached acquisition worker running; later recovery retained the same acquisition identifier. Development-terminal process-tree cancellation was separately observed to interrupt an acquisition and is documented honestly.

The shared CLI replayed the GUI's exact admitted comparison request with its idempotency key while it was active and returned the same job without launching another worker. That preceding-cohort parity receipt is retained outside Git in the independent runtime's `checks\operator-evidence\cli-parity.json`. On the final source, the exact step-three train request/key was replayed after its yielded state: the GUI and CLI returned the same `52d639d5dec74f9a90840cc58b36c133`, the project job count stayed at three and its retained PID stayed unchanged. The final-source receipt is `checks\operator-evidence\final-source-cli-parity.json`. This verifies shared durable records and idempotence, not a separate trainer.

A preceding comparison worker disappeared after partial evaluation; no reliable exception or Windows crash event established its cause. The interruption remains recorded. An explicit same-candidate retry completed, with an observed process exit code 0. Independent review subsequently found two actual reliability defects: a directory entry could vanish between metadata reads during storage accounting, and failed progress publication could prevent the durable failure reason from being saved. Both were repaired and covered by targeted tests. These defects are not asserted to be the proven cause of the unexplained disappearance.

The final cohort used the unchanged demo generator, recipe and generation settings after those source-identity repairs. It is a software-workflow rerun, not a new independent generalization study. Earlier results and failed/interrupted attempts were preserved; no checkpoint or recipe was selected using the final-test scores.

An actual transient Windows progress-file access error was also retained. The worker completed and the receipt was subsequently readable. The GUI now shows a meaningful temporary-unavailability response and directs the operator to inspect the existing job before retrying. The underlying transient OS-access cause was not proven. A targeted check verifies 503 followed by a successful refresh without changing the completed job or starting a retry.

The final GUI case-review check discovered a route-order defect: the generic job-control POST route shadowed the fixed assessment POST route. It returned a truthful 400 and stored no review. A regression test reproduced that failure before the repair. Fixed routes now precede the generic route, following [FastAPI's documented route-order rule](https://fastapi.tiangolo.com/tutorial/path-params/#order-matters). The repaired GUI saved the delegated assessment; the CLI retrieved the identical record. Tests also verify GET persistence, wrong-project/case rejection and refusal of unsupported recovery actions. Independent source review closed this finding. The source identity changed, so the final cohort above was admitted anew with the same recipe rather than rewriting earlier receipts.

The preceding complete cohort under source `5cbbfe832c7f9bcc12e1696c6c07a6fcf21229c58e00268b5da3211c3b049969` remains retained in project `27191efcdb37404abd492579cc7bdc67`: comparison `83fe9bb9490f40de9e5a9f786e9402a8` completed with 8/16 baseline successes and 11/16 candidate successes, 5 to 0 invalid outputs, 3 to 7 unsupported-field proxy counts, no model-call failures or case regressions, and failed declared criteria. Export `951a2add265147d0a2c7f4172fe4ae49` actually loaded the package in an independent `python -I` Transformers/PEFT process, reproduced the retained candidate probe, recorded `no_lab_imports: true`, `quality_acceptance: false` and `activated: false`. This is preserved prior-source proof, not a substitute for the final-source run.

During the final-source baseline, a poll again encountered transient access denial to the mutable progress receipt. The worker continued and completed; the actual GUI Refresh recovered the same job `692bd5023ce24088a14f18476a64de1b`, displayed its completed state and cleared the temporary message. No new baseline or automatic model retry was created. This records observed recovery while the underlying Windows access cause remains unresolved.

## Software checks and review

- Full local CPU suite on the repaired source: 115 tests passed in approximately 36 seconds, with one development-only Starlette/httpx deprecation warning. Three affected route/security/recovery tests also passed separately after the initially failing route reproduction.
- Ruff passed across source and tests; whitespace checks passed. No hosted CI result is asserted.
- Independent Codex review found material issues, which were repaired and verified. The final source review reported no remaining material findings. The reviewer did not run the model demonstration; its source review and this agent's runtime evidence are separate proof stages.
- The full 51-package CPU lock resolved afresh for Windows x64 CPython 3.12. It uses the exact official CPU torch wheel/hash to avoid an index-shadowing conflict. The running environment was not replaced by a clean reinstall.

## Original-project preservation

Read-only verification retained the original Lab at `7f55e77c7bd67a2d2eae9164b6ce7f57aa6dd8d5` with its preexisting modified `AGENTS.md` and untracked `PROJECT.md`. The installed original `training-lab.json` remained 3,092 bytes with SHA-256 `4c22e055d410f9a2b5a917df4ad7ea8bbf717171561d56bf38dd31aeb3f8f74e` and last modification `2026-10-04T04:01:58.5957218Z`, before this lane. RAW made no application/configuration/Git writes to those original projects. Q-Trades was clean on `main` at `65fc680`; it advanced independently during this lane through PR #63, so an unchanged Q-Trades HEAD is not claimed.

## Capability and publication limits

The demonstrated fast-task profile is Qwen3-0.6B non-thinking CPU LoRA. Tev `tev1:4b` remains the user's preferred System 1 model; `tev1:0.8b` remains experimental. Read-only installed metadata confirms those GGUF artifacts, but trainable source, weight release terms and an approved local resource envelope are still gates. No Tev inference/training benchmark, BexDog change, pretrained reasoning-model run, GPU/QLoRA execution, non-generative classifier/reranker training or GRPO proof is claimed. Tiny inherited hybrid-model tests establish software behavior only.

The measured source is committed on `codex/operator-workflow` in the managed implementation worktree. The new repository's initial README history is preserved as an ancestor; there was no forced remote replacement. On 2026-10-06, the owner explicitly authorized keeping SAC-CS112-Randall-Christopher/RAW_Training_Labs public and uploading a source-only copy, superseding the earlier private-source requirement. The publication copy uses completed commit `85e28d5841127293e6ec9da79f2e889df7b827b8` plus publication-documentation updates; unfinished desktop changes remain in their active development lane. This authorization covers source and documentation, not operational data, model artifacts, deployments or activation. Keep the implementation worktree because the editable runtime and ongoing desktop work still require it. The root checkout remains clean on its preserved foundation `main`; it has not been force-reset or described as synchronized with the remote's bootstrap branch.

## Remaining decisions and continuation

1. The owner resolved the privacy choice on 2026-10-06 by explicitly authorizing public source publication. Refresh the remote branch and preserve any new remote changes before the source-only push/draft pull request. Keep unfinished desktop changes in their existing lane. Do not merge, deploy or activate a model as part of publication.
2. Keep the failed-quality candidate and its runtime package as development evidence. A subsequent quality iteration requires a fresh independently held-out population; do not reuse the exposed final cases to select another recipe/checkpoint or claim new generalization evidence.
3. Establish Tev's exact trainable release/rights and a separately approved resource profile before executable Tev training. Real reasoning, QLoRA and non-generative backends require their own measured checkpoints; GRPO remains later.
4. After a durable reviewed outcome and when this worktree is no longer needed by the installed editable runtime/review, retire it through the managed worktree workflow. Preserve private operational evidence separately and synchronize the root baseline safely. There are no extra abandoned review/implementation worktrees to remove in this lane.

See [open-source-review.md](open-source-review.md) for the primary-source investigation and reuse decisions, [operator-runbook.md](operator-runbook.md) for operation/recovery, and [model-coverage.md](model-coverage.md) for honest model-specific gates.
