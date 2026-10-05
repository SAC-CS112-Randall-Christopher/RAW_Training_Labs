"""Detached, owned and bounded job execution; the browser is not its lifetime owner."""

import argparse
import os
import threading
import time

import psutil
from llm_lab.corpus import verify_corpus
from llm_lab.io import read_object, sha_file, write_state
from llm_lab.model import environment, verify_model
from llm_lab.resources import low_priority

from raw_training_labs.jobs import source_identity
from raw_training_labs.profiles import recipe_for, supported
from raw_training_labs.service import Service


def force_stop(store, process, directory, job_id, reason, peak, *, status="failed", code=70):
    """Stop owned runtime children and exit even when durable reporting is unavailable."""
    try:
        try:
            children = process.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                try:
                    child.kill()
                except psutil.Error:
                    pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            write_state(
                directory / "progress.json",
                {
                    "phase": status,
                    "reason": reason,
                    "peak_owned_rss_bytes": peak,
                    "forced_stop": True,
                },
            )
        except Exception:
            pass
        try:
            with store.transaction() as db:
                db.execute("UPDATE jobs SET status=?,reason=? WHERE id=?", (status, reason, job_id))
        except Exception:
            # Recovery reconciles an absent worker as interrupted, never successful.
            pass
    finally:
        os._exit(code)


def execute(root, project_id, job_id):
    started = time.monotonic()
    service = Service(root)
    store = service.store
    job = store.job(project_id, job_id)
    request = job["request"]
    directory = store.artifact_path(project_id, f"jobs/{job_id}")
    identity = read_object(directory / "identity.json")
    if identity["source"] != source_identity():
        raise ValueError("Source changed between admission and worker launch")
    profile = supported(request["profile_id"])
    recipe = recipe_for(request["profile_id"])
    base = store.artifact_path(project_id, f"models/{profile['id']}")
    corpus = frozen = manifest = None
    if environment() != identity["preflight"]["environment"]:
        raise ValueError("Admitted environment changed")
    process = psutil.Process()
    done = threading.Event()
    peak = [0]
    with store.transaction() as db:
        db.execute(
            "UPDATE jobs SET status='running',pid=?,process_created=?,heartbeat=? WHERE id=?",
            (os.getpid(), process.create_time(), time.time(), job_id),
        )
    low_priority()

    def monitor():
        while not done.wait(1):
            try:
                rss = process.memory_info().rss + sum(
                    p.memory_info().rss for p in process.children(recursive=True) if p.is_running()
                )
                peak[0] = max(peak[0], rss)
                with store.transaction() as db:
                    db.execute("UPDATE jobs SET heartbeat=? WHERE id=?", (time.time(), job_id))
                reason = None
                if rss > profile["rss_limit_bytes"]:
                    reason = "Approved 8 GiB owned-worker RSS ceiling exceeded"
                deadline = (
                    recipe.max_run_seconds
                    if request["kind"] == "train"
                    else 600
                    if request["kind"] == "acquire"
                    else profile["wall_limit_seconds"]
                )
                if time.monotonic() - started > deadline:
                    reason = "Approved job wall-time ceiling exceeded"
                if int(time.monotonic() - started) % 10 == 0:
                    store.budget()
                if (directory / "cancel.request").exists() and request["kind"] == "train":
                    (directory / "yield.request").touch(exist_ok=True)
                if (directory / "cancel.request").exists() and request["kind"] == "acquire":
                    force_stop(
                        store,
                        process,
                        directory,
                        job_id,
                        "Acquisition cancelled; partial download retained",
                        peak[0],
                        status="cancelled",
                        code=72,
                    )
                if reason:
                    force_stop(store, process, directory, job_id, reason, peak[0])
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                # Resource monitoring failures cannot silently remove the safety envelope.
                force_stop(
                    store,
                    process,
                    directory,
                    job_id,
                    f"Resource monitor unavailable: {exc}",
                    peak[0],
                    code=71,
                )

    watchdog = threading.Thread(target=monitor, daemon=True)
    watchdog.start()
    status, reason, result = "failed", None, None

    def progress(phase, **details):
        write_state(
            directory / "progress.json",
            {
                "phase": phase,
                "elapsed_seconds": time.monotonic() - started,
                "peak_owned_rss_bytes": peak[0],
                **details,
            },
        )

    def cancelled():
        return (directory / "cancel.request").exists()

    try:
        if cancelled():
            raise RuntimeError("Job cancelled before execution")
        if request["kind"] != "acquire":
            progress("Verifying frozen base and corpus identities")
            corpus = store.artifact(project_id, request["corpus_id"], "corpus")
            frozen = verify_model(base)
            manifest, _ = verify_corpus(corpus["path"], splits=())
            if (
                frozen["base_sha256"] != identity["preflight"]["base_sha256"]
                or manifest["corpus_sha256"] != identity["preflight"]["corpus_sha256"]
            ):
                raise ValueError("Admitted base or corpus changed")
        if request["kind"] == "acquire":
            from raw_training_labs.acquire import acquire

            progress("Acquiring explicitly authorized pinned model; no training")
            result = acquire(service, project_id, profile["id"], authorized=True)
            status = "completed"
        elif request["kind"] in {"baseline", "compare"}:
            from raw_training_labs.evaluate import arm
            from raw_training_labs.scoring import comparison

            splits = (
                ("validation", "regression")
                if request["kind"] == "baseline"
                else ("test", "regression")
            )
            _, files = verify_corpus(corpus["path"], splits=splits)
            criteria = manifest["criteria"]
            progress(
                "Loading configured baseline", expected_cases=sum(len(v) for v in files.values())
            )
            baseline = arm(
                base,
                recipe,
                profile,
                files,
                criteria,
                stop=cancelled,
                progress=lambda cases: progress("Evaluating baseline", cases=cases),
            )
            write_state(directory / "baseline.json", baseline)
            if baseline["loading_error"]:
                raise RuntimeError("Baseline loading failed; all affected cases retained")
            result = {
                "baseline": baseline,
                "profile": profile,
                "criteria": criteria,
                "corpus_sha256": manifest["corpus_sha256"],
                "base_sha256": frozen["base_sha256"],
                "environment": environment(),
            }
            if request["kind"] == "compare":
                run = store.artifact_path(project_id, f"jobs/{request['candidate_id']}/run")
                receipt = read_object(run / "run.json")
                if receipt["status"] != "trained" or receipt["recipe"] != recipe.model_dump():
                    raise ValueError(
                        "Selected candidate did not complete the identical training profile"
                    )
                if (
                    receipt["base_sha256"] != frozen["base_sha256"]
                    or receipt["corpus_sha256"] != manifest["corpus_sha256"]
                    or receipt["environment"] != environment()
                ):
                    raise ValueError(
                        "Candidate's retained training identities differ from evaluation"
                    )
                for name, expected in receipt["candidate_files"].items():
                    if sha_file(run / "best-adapter" / name) != expected:
                        raise ValueError("Candidate adapter changed before comparison")
                candidate = arm(
                    base,
                    recipe,
                    profile,
                    files,
                    criteria,
                    adapter=run / "best-adapter",
                    stop=cancelled,
                    progress=lambda cases: progress("Evaluating candidate", cases=cases),
                )
                write_state(directory / "candidate.json", candidate)
                result["comparison"] = comparison(baseline["cases"], candidate["cases"], criteria)
                result["candidate"] = candidate
                result["sealed_final_evaluation"] = True
                result["no_test_checkpoint_selection"] = True
                if candidate["loading_error"]:
                    raise RuntimeError(
                        "Candidate loading failed; complete failure denominator retained"
                    )
            status = "completed"
        elif request["kind"] == "train":
            from llm_lab.training import train

            progress("Preparing adapter training; only train/validation labels are read")
            resume = (
                store.artifact_path(project_id, f"jobs/{request['resume_id']}/run")
                if request["resume_id"]
                else None
            )
            result = train(
                base,
                corpus["path"],
                recipe,
                directory / "run",
                resume=resume,
                yield_file=directory / "yield.request",
                yield_after_step=request["yield_after_step"],
            )
            status = {"trained": "completed", "yielded": "yielded", "deferred": "deferred"}.get(
                result["status"], "failed"
            )
            reason = result.get("reason")
        elif request["kind"] == "export":
            from raw_training_labs.delivery import export

            progress("Packaging selected PEFT adapter and checking independent runtime")
            result = export(service, project_id, request, directory, profile)
            status = "completed"
        if cancelled():
            status, reason = (
                "cancelled",
                "Operator cancellation; inspect checkpoint availability explicitly",
            )
        verify_model(base)
        store.budget()
        if result is not None:
            result["worker_resources"] = {
                "wall_seconds": time.monotonic() - started,
                "peak_owned_rss_bytes": peak[0],
                "cpu_threads": recipe.cpu_threads,
            }
            write_state(directory / "result.json", result)
        progress(status, reason=reason)
    except BaseException as exc:
        status = "cancelled" if cancelled() else "failed"
        reason = f"{type(exc).__name__}: {exc}"
        progress(
            status,
            reason=reason,
            next_action="Inspect retained error and receipts; retry only after "
            "correcting the cause",
        )
    finally:
        done.set()
        watchdog.join(timeout=2)
        with store.transaction() as db:
            db.execute(
                "UPDATE jobs SET status=?,reason=?,heartbeat=? WHERE id=?",
                (status, reason, time.time(), job_id),
            )
    return 0 if status in {"completed", "yielded", "deferred", "cancelled"} else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--job", required=True)
    arguments = parser.parse_args()
    try:
        raise SystemExit(execute(arguments.root, arguments.project, arguments.job))
    except Exception as error:
        # Even failures before model loading retain a durable, useful outcome.
        from raw_training_labs.storage import Store

        with Store(arguments.root).transaction() as connection:
            connection.execute(
                "UPDATE jobs SET status='failed',reason=? WHERE id=? AND project=?",
                (
                    f"Worker initialization failed: {type(error).__name__}: {error}",
                    arguments.job,
                    arguments.project,
                ),
            )
        raise
