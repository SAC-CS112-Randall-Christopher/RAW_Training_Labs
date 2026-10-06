"""Durable single-job admission, fixed owned processes, idempotence and recovery."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from llm_lab.io import canonical, digest, read_object, sha_file, write_new

from raw_training_labs.contracts import Start
from raw_training_labs.profiles import supported
from raw_training_labs.storage import identifier, new_id

ACTIVE = ("starting", "running", "cancelling")


def source_identity():
    roots = [Path(__file__).parent, Path(__file__).parent.parent / "llm_lab"]
    files = {f"{r.name}/{p.name}": sha_file(p) for r in roots for p in sorted(r.glob("*.py"))}
    return {"files": files, "sha256": digest(files)}


def process_alive(job):
    if job["pid"] is None or job["process_created"] is None:
        return False
    try:
        process = psutil.Process(job["pid"])
        args = process.cmdline()
        return abs(process.create_time() - job["process_created"]) < 0.1 and (
            "raw_training_labs.worker" in args and job["id"] in args and process.is_running()
        )
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return None


def reconcile(store):
    now = time.time()
    with store.transaction() as db:
        rows = db.execute(
            "SELECT * FROM jobs WHERE status IN ('starting','running','cancelling')"
        ).fetchall()
        for row in rows:
            job = dict(row)
            # A launch transaction intentionally precedes Popen. Do not race a new launch.
            if row["status"] == "starting" and now - row["created"] < 30:
                continue
            if process_alive(job) is False:
                db.execute(
                    "UPDATE jobs SET status='interrupted',reason=? WHERE id=?",
                    (
                        "Owned worker disappeared before a complete result. "
                        "No clean checkpoint is implied.",
                        row["id"],
                    ),
                )


def jobs(store, project_id):
    store.project(project_id)
    reconcile(store)
    with store.connect() as db:
        rows = db.execute(
            "SELECT id FROM jobs WHERE project=? ORDER BY created DESC LIMIT 100", (project_id,)
        ).fetchall()
    return [detail(store, project_id, r["id"]) for r in rows]


def detail(store, project_id, job_id):
    job = store.job(project_id, job_id)
    root = store.artifact_path(project_id, f"jobs/{job_id}")
    result = {k: v for k, v in job.items() if k != "key"}
    result["output_location"] = str(root)
    for name in ("progress.json", "result.json"):
        if (root / name).is_file():
            result[name.removesuffix(".json")] = read_object(root / name)
    if (root / "run/run.json").is_file():
        receipt = read_object(root / "run/run.json")
        result["training"] = {
            k: receipt.get(k)
            for k in (
                "status",
                "step",
                "best_step",
                "history",
                "reason",
                "wall_seconds",
                "trainable_parameters",
                "profile_sha256",
            )
        }
        result["resume_supported"] = (
            receipt.get("status") == "yielded" and (root / "run/checkpoint.json").is_file()
        )
    return result


def completed(store, project_id, job_id, kind):
    job = store.job(project_id, job_id)
    if job["status"] != "completed" or job["kind"] != kind:
        raise ValueError(f"Require a completed {kind} job from this project")
    return job


def _same(request, other):
    if request.corpus_id != other["corpus_id"] or request.profile_id != other["profile_id"]:
        raise ValueError("Comparison/recovery requires identical corpus and model profile")


def match_identity(store, project_id, job_id, current):
    path = store.artifact_path(project_id, f"jobs/{job_id}/identity.json")
    recorded = read_object(path)
    expected = recorded["preflight"]
    fields = ("base_sha256", "corpus_sha256", "recipe", "masking", "environment", "profile")
    if any(expected.get(k) != current.get(k) for k in fields):
        raise ValueError("Retained job base, data, recipe, labels, mode or environment differs")
    if recorded["source"] != source_identity():
        raise ValueError(
            "Retained job source differs; evaluate a new configured baseline explicitly"
        )


def start(service, project_id, payload):
    request = Start.model_validate(payload)
    store = service.store
    store.project(project_id)
    if request.kind != "acquire":
        identifier(request.corpus_id)
    profile = supported(request.profile_id)
    if not request.authorized:
        raise ValueError("Explicit authorization is required for every model job")
    reconcile(store)
    serialized = request.model_dump()
    with store.connect() as db:
        prior = db.execute(
            "SELECT id,request FROM jobs WHERE project=? AND key=?",
            (project_id, request.idempotency_key),
        ).fetchone()
    if prior:
        if json.loads(prior["request"]) != serialized:
            raise ValueError("Idempotency key already belongs to a different request")
        return detail(store, project_id, prior["id"])
    if request.kind == "acquire":
        from llm_lab.model import environment

        from raw_training_labs.profiles import recipe_for

        preflight = {
            "eligible": True,
            "profile": profile,
            "recipe": recipe_for(request.profile_id).model_dump(),
            "environment": environment(),
            "storage": store.budget(reserve=3 * 1024**3),
        }
        if any((request.baseline_id, request.candidate_id, request.resume_id, request.corpus_id)):
            raise ValueError("Acquisition uses only the explicitly selected pinned public model")
    else:
        preflight = service.preflight(project_id, request.corpus_id, request.profile_id)
    if not preflight["eligible"]:
        raise ValueError(preflight["reason"])
    if request.kind in {"train", "compare", "export"}:
        if request.baseline_id is None:
            raise ValueError("A completed configured baseline is required before candidate work")
        baseline = completed(store, project_id, request.baseline_id, "baseline")
        _same(request, baseline["request"])
        match_identity(store, project_id, baseline["id"], preflight)
    if request.kind == "train":
        with store.connect() as db:
            exposed = db.execute(
                "SELECT request FROM jobs WHERE project=? AND kind='compare'", (project_id,)
            ).fetchall()
        if any(json.loads(r["request"])["corpus_id"] == request.corpus_id for r in exposed):
            raise ValueError(
                "Final evaluation has been admitted for this corpus; further training "
                "requires a fresh independently reviewed population"
            )
    if request.kind in {"compare", "export"}:
        if request.candidate_id is None:
            raise ValueError("Select a completed candidate from this project")
        candidate = completed(store, project_id, request.candidate_id, "train")
        _same(request, candidate["request"])
        match_identity(store, project_id, candidate["id"], preflight)
        if candidate["request"]["baseline_id"] != request.baseline_id:
            raise ValueError("Candidate was trained against a different baseline")
    if request.kind == "export":
        with store.connect() as db:
            comparisons = db.execute(
                "SELECT request FROM jobs WHERE project=? AND kind='compare' "
                "AND status='completed'",
                (project_id,),
            ).fetchall()
        if not any(
            json.loads(r["request"])["candidate_id"] == request.candidate_id
            and json.loads(r["request"])["baseline_id"] == request.baseline_id
            for r in comparisons
        ):
            raise ValueError("Complete the matched comparison before export")
    if request.resume_id:
        if request.kind != "train":
            raise ValueError("Only training supports checkpoint resume")
        previous = store.job(project_id, request.resume_id)
        if previous["kind"] != "train" or previous["status"] not in {"yielded", "cancelled"}:
            raise ValueError("Resume requires an owned cleanly yielded training attempt")
        _same(request, previous["request"])
        if previous["request"]["baseline_id"] != request.baseline_id:
            raise ValueError("Resume baseline changed")
        previous_root = store.artifact_path(project_id, f"jobs/{request.resume_id}")
        if read_object(previous_root / "identity.json")["source"] != source_identity():
            raise ValueError("Worker source changed; exact checkpoint recovery is not compatible")
        if not read_object(previous_root / "run/checkpoint.json").get("resume_safe"):
            raise ValueError("No safe optimizer checkpoint; start a new candidate deliberately")
    if request.yield_after_step is not None and request.kind != "train":
        raise ValueError("Step yielding is a training-only action")
    job_id = new_id()
    with store.transaction() as db:
        # Recheck identity under the admission lock, including races during expensive preflight.
        prior = db.execute(
            "SELECT id,request FROM jobs WHERE project=? AND key=?",
            (project_id, request.idempotency_key),
        ).fetchone()
        if prior:
            if json.loads(prior["request"]) != serialized:
                raise ValueError("Idempotency key conflict")
            return detail(store, project_id, prior["id"])
        if db.execute(
            "SELECT 1 FROM jobs WHERE status IN ('starting','running','cancelling')"
        ).fetchone():
            raise ValueError("One heavy RAW job is already active; wait, yield or cancel it")
        if request.kind == "train":
            exposed = db.execute(
                "SELECT request FROM jobs WHERE project=? AND kind='compare'", (project_id,)
            ).fetchall()
            if any(json.loads(r["request"])["corpus_id"] == request.corpus_id for r in exposed):
                raise ValueError("Final evaluation already admitted; further training is blocked")
        if db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= 10000:
            raise ValueError("Job history limit reached; explicit archival is required")
        directory = store.artifact_path(project_id, f"jobs/{job_id}")
        directory.mkdir(parents=True, exist_ok=False)
        write_new(
            directory / "identity.json",
            {
                "source": source_identity(),
                "preflight": preflight,
                "project_id": project_id,
                "job_id": job_id,
            },
        )
        db.execute(
            "INSERT INTO jobs(id,project,kind,status,request,key,created) VALUES(?,?,?,?,?,?,?)",
            (
                job_id,
                project_id,
                request.kind,
                "starting",
                canonical(serialized),
                request.idempotency_key,
                time.time(),
            ),
        )
    args = [
        sys.executable,
        "-m",
        "raw_training_labs.worker",
        "--root",
        str(store.root),
        "--project",
        project_id,
        "--job",
        job_id,
    ]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(Path(__file__).parent.parent),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
    )
    if request.kind == "acquire":
        env.update(
            HF_HUB_OFFLINE="0",
            TRANSFORMERS_OFFLINE="0",
            HF_HUB_DISABLE_XET="1",
            HF_HUB_DOWNLOAD_TIMEOUT="30",
            HF_HUB_ETAG_TIMEOUT="15",
            HF_HOME=str(store.root / "cache"),
        )
    process = None
    try:
        with (directory / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                cwd=directory,
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
                if os.name == "nt"
                else 0,
                start_new_session=os.name != "nt",
            )
        created = psutil.Process(process.pid).create_time()
        with store.transaction() as db:
            db.execute(
                "UPDATE jobs SET pid=?,process_created=? WHERE id=?", (process.pid, created, job_id)
            )
        # Worker records its own creation identity too. Parent's Popen is deliberately detached.
    except BaseException as exc:
        # A successful Popen followed by failed identity persistence still owns a live worker.
        # Do not release single-job admission until that launched process is stopped.
        if process is not None and process.poll() is None:
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:
                children = []
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait(timeout=5)
        with store.transaction() as db:
            db.execute(
                "UPDATE jobs SET status='failed',reason=? WHERE id=? "
                "AND status IN ('starting','running','cancelling')",
                (f"Worker launch failed: {exc}", job_id),
            )
        raise
    return detail(store, project_id, job_id)


def cancel(store, project_id, job_id):
    job = store.job(project_id, job_id)
    if job["status"] not in ACTIVE:
        raise ValueError("Job is not active")
    directory = store.artifact_path(project_id, f"jobs/{job_id}")
    (directory / "cancel.request").touch(exist_ok=True)
    with store.transaction() as db:
        db.execute(
            "UPDATE jobs SET status='cancelling' WHERE id=? "
            "AND status IN ('starting','running','cancelling')",
            (job_id,),
        )
    return detail(store, project_id, job_id)


def yield_job(store, project_id, job_id):
    job = store.job(project_id, job_id)
    if job["kind"] != "train" or job["status"] not in ACTIVE:
        raise ValueError("Only an active training job can yield to a checkpoint")
    (store.artifact_path(project_id, f"jobs/{job_id}") / "yield.request").touch(exist_ok=True)
    return detail(store, project_id, job_id)
