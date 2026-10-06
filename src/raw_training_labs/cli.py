"""Matching operator CLI with explicit projects, JSON results and reliable failures."""

import argparse
import json
import sys
import time
from pathlib import Path

from raw_training_labs.service import Service
from raw_training_labs.storage import DEFAULT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="raw-labs", description="RAW local training/evaluation workbench"
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="Owned private RAW storage outside source"
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("projects", help="List projects and stable identifiers")
    sub.add_parser("profiles", help="Supported capabilities and honest coverage limits")
    p = sub.add_parser("project-create", help="Create task and criteria from a JSON object")
    p.add_argument("file", type=Path)
    for name in ("summary", "examples", "validate", "prepare", "jobs", "artifacts", "demo-review"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True)
    p = sub.add_parser("import", help="Import neutral examples; originals remain pending review")
    p.add_argument("--project", required=True)
    p.add_argument("file", type=Path)
    p = sub.add_parser("review")
    p.add_argument("--project", required=True)
    p.add_argument("--example", required=True)
    p.add_argument("file", type=Path)
    p = sub.add_parser("acquire", help="Explicit pinned model download; never part of bootstrap")
    p.add_argument("--project", required=True)
    p.add_argument("--profile", default="qwen3-0.6b-fast-cpu")
    p.add_argument("--authorized", action="store_true")
    p.add_argument("--key", help="Stable request key for an idempotent acquisition retry")
    p = sub.add_parser("preflight")
    p.add_argument("--project", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--profile", default="qwen3-0.6b-fast-cpu")
    p = sub.add_parser(
        "start", help="Start a baseline/train/compare/export job from an explicit JSON request"
    )
    p.add_argument("--project", required=True)
    p.add_argument("file", type=Path)
    for name in ("job", "cancel", "yield", "wait"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True)
        p.add_argument("--job", required=True)
        if name == "wait":
            p.add_argument("--seconds", type=int, default=60)
    for name in ("assess", "assessments"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True)
        p.add_argument("--job", required=True)
        if name == "assess":
            p.add_argument("file", type=Path)
    sub.add_parser("demo-create", help="Create unreviewed RAW synthetic examples; no model load")
    p = sub.add_parser("serve", help="Launch local-only GUI; no training or acquisition")
    p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        service = Service(args.root)
        op = args.operation
        from llm_lab.io import read_object

        from raw_training_labs import jobs

        match op:
            case "projects":
                result = service.projects()
            case "profiles":
                result = service.profiles()
            case "project-create":
                result = service.create_project(read_object(args.file))
            case "summary":
                result = service.summary(args.project)
            case "examples":
                result = service.examples(args.project)
            case "validate":
                result = service.validate(args.project)
            case "prepare":
                result = service.prepare(args.project)
            case "artifacts":
                result = service.artifacts(args.project)
            case "jobs":
                result = jobs.jobs(service.store, args.project)
            case "import":
                if args.file.suffix.lower() == ".jsonl":
                    from llm_lab.io import read_rows

                    payload = {"examples": read_rows(args.file)}
                else:
                    payload = read_object(args.file)
                result = service.import_examples(args.project, payload)
            case "review":
                result = service.review(args.project, args.example, read_object(args.file))
            case "acquire":
                import uuid

                result = jobs.start(
                    service,
                    args.project,
                    {
                        "kind": "acquire",
                        "profile_id": args.profile,
                        "authorized": args.authorized,
                        "idempotency_key": args.key or str(uuid.uuid4()),
                    },
                )
            case "preflight":
                result = service.preflight(args.project, args.corpus, args.profile)
            case "start":
                result = jobs.start(service, args.project, read_object(args.file))
            case "job":
                jobs.reconcile(service.store)
                result = jobs.detail(service.store, args.project, args.job)
            case "cancel":
                result = jobs.cancel(service.store, args.project, args.job)
            case "yield":
                result = jobs.yield_job(service.store, args.project, args.job)
            case "assess":
                result = service.assess(args.project, args.job, read_object(args.file))
            case "assessments":
                result = service.assessments(args.project, args.job)
            case "wait":
                if not 1 <= args.seconds <= 7200:
                    raise ValueError("Wait budget must be 1-7200 seconds")
                deadline = time.monotonic() + args.seconds
                while True:
                    jobs.reconcile(service.store)
                    result = jobs.detail(service.store, args.project, args.job)
                    if result["status"] not in jobs.ACTIVE or time.monotonic() >= deadline:
                        break
                    time.sleep(1)
                if result["status"] not in {"completed", "yielded"}:
                    print(json.dumps(result, ensure_ascii=False))
                    return 3 if result["status"] in jobs.ACTIVE else 2
            case "demo-create":
                from raw_training_labs.demo import create

                result = create(service)
            case "demo-review":
                from raw_training_labs.demo import review_synthetic

                result = review_synthetic(service, args.project)
            case "serve":
                if not 1024 <= args.port <= 65535:
                    raise ValueError("Choose an unprivileged local port")
                import uvicorn

                from raw_training_labs.server import application

                uvicorn.run(
                    application(args.root, args.port),
                    host="127.0.0.1",
                    port=args.port,
                    access_log=False,
                )
                return 0
            case _:
                raise ValueError("Unsupported operation")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, default=str))
        return 0
    except (ValueError, OSError, RuntimeError, TypeError, ModuleNotFoundError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "next_action": "Correct the stated input or resource issue "
                    "and retry deliberately.",
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
