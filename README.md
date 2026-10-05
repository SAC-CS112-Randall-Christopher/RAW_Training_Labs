# RAW Training Labs

Independent local model-development workbench for Chris Randall / Randall Automation Works LLC. The GUI and CLI use the same project, review, frozen-data and durable-job records. No original Q-Trades Lab environment, dataset, weight, operating configuration or Git history is reused.

The first supported execution profile is the approved Qwen3-0.6B CPU supervised LoRA demonstration. Tev1 4B is the preferred System 1 profile; its installed GGUF identity and decision format are recorded, while its trainable-source rights, local compatibility and resource approval remain separate gates. Thinking and non-thinking behavior are explicit capability profiles. Inspect **Model coverage** for honest supported, gated and reference-only states.

## Start the local workbench

Use a separate Python 3.12 environment outside all source checkouts. The verified Windows x64 CPU environment is captured in `requirements-cpu.lock`, including the exact official PyTorch CPU wheel and its hash. Other platforms need a separately validated environment. The inherited requirement files remain provenance, not the new workbench's installation authority.

```powershell
uv venv 'C:\Projects\RAW-Training-Labs-Runtime\cpu' --python 3.12
uv pip install --python 'C:\Projects\RAW-Training-Labs-Runtime\cpu\Scripts\python.exe' -r requirements-cpu.lock
uv pip install --python 'C:\Projects\RAW-Training-Labs-Runtime\cpu\Scripts\python.exe' --no-deps -e .
& 'C:\Projects\RAW-Training-Labs-Runtime\cpu\Scripts\raw-labs.exe' serve --port 8765
```

Open http://127.0.0.1:8765/. Startup creates/opens only the owned RAW private store; it does not acquire a model, train, use paid compute, publish data, install a service or activate an adapter. The default data root is `G:\Projects\RAW-Training-Labs-Data`. A different explicitly owned location can be selected with `raw-labs --root <absolute-path> ...`; it must be outside source and may not adopt an unrelated nonempty directory.

## Operator workflow

1. Create a project with its customer task, initial approach and declared success criteria. A configured model, documents/retrieval, tools or a combination may solve the task before training.
2. Import original JSON/JSONL examples with explicit source, group, split and permissions. Review teaching targets, corrections, reviewer identity/type and actual use rights; originals and revisions remain available. Synthetic author review is clearly labeled and does not stand in for independent human/customer review.
3. Validate and freeze reviewed data. Rights, groups, conflicts, chronology and existing exposure can block preparation. Final targets are sealed after freeze; subsequent validation does not inspect them.
4. Select a supported behavior profile, explicitly acquire its pinned public model if needed, and inspect data/hardware preflight. The initial profile uses two CPU threads and an 8 GiB owned-worker memory ceiling; GPU execution is gated separately.
5. Authorize and evaluate the configured baseline before training. Every failed/invalid call remains in the denominator. Only a compatible completed baseline permits a candidate.
6. Authorize training. Watch durable progress, logs, successful steps and output locations. The bounded demo can yield after step three and resume to six in a new attempt. Resume requires a verified clean checkpoint with identical base, data, recipe, labels, mode, source and environment.
7. Authorize the matched final comparison. Compare exact task fields, invalid outputs, unsupported field claims, regressions, call failures, latency and observed memory. Final-test evidence cannot select another checkpoint or a later training attempt from the same frozen population.
8. Review disagreements and authorize a sensitive development export. The package includes PEFT adapter, tokenizer, exact identities, recipe, evaluation, source license and limitations. Independent runtime verification loads it in a separate process. Customer acceptance and installation/activation remain separate.

Only one heavy RAW job can be admitted across all projects. Closing the browser or terminating only the web server does not own the worker's lifetime. Abruptly stopping a whole process tree or the machine may interrupt a worker; reconciliation never invents success or a safe checkpoint.

See [the operator runbook](docs/operator-runbook.md), [the approved project boundaries](PROJECT.md), [source-only import provenance](docs/source-import.json), [the open-source investigation](docs/open-source-review.md) and [the measured local acceptance record](docs/acceptance-record.md). Measured acceptance is recorded separately from software checks.

## Matching CLI

Every project/job command requires its explicit stable identifier. JSON requests use the same validated contracts as the GUI.

```powershell
raw-labs projects
raw-labs profiles
raw-labs summary --project <project-id>
raw-labs validate --project <project-id>
raw-labs prepare --project <project-id>
raw-labs acquire --project <project-id> --profile qwen3-0.6b-fast-cpu --authorized --key <unique-stable-key>
raw-labs preflight --project <project-id> --corpus <corpus-id>
raw-labs start --project <project-id> <request.json>
raw-labs job --project <project-id> --job <job-id>
raw-labs wait --project <project-id> --job <job-id> --seconds 60
raw-labs yield --project <project-id> --job <job-id>
raw-labs cancel --project <project-id> --job <job-id>
raw-labs assess --project <project-id> --job <comparison-id> <review.json>
```

Exit `0` means the requested operation completed, `2` means an input/resource/terminal failure, and `3` means a bounded wait ended while the job remains active. An admitted job is not necessarily a completed or quality-accepted model. Use `--help` for review/import/artifact and synthetic-demo commands.

## Verification and scope

```powershell
python -m pytest -q --basetemp 'C:\Projects\RAW-Training-Labs-Runtime\checks\unique-run'
python -m ruff check src tests
```

Tiny randomly initialized fixtures prove training/masking/checkpoint/software behavior; they do not prove pretrained reasoning, improvement, customer utility or deployment. The initial GUI/CLI supports a bounded compatible generative LoRA profile. QLoRA, additional models and context/resource envelopes require separate compatibility evidence. Non-generative classifiers/rerankers, GRPO/RL, distributed/full-model training, GGUF conversion and customer activation are not silently supported.

Keep all operational data, caches, downloads, checkpoints, model packages and temporary exports outside Git and source checkouts. RAW source remains private under the approved project brief; upstream project/package/model licenses are distinct from RAW's source ownership.
