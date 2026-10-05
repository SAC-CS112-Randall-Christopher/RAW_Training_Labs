"""Outcome checks for shared project, review, job, local-request and scoring boundaries."""

import copy
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from llm_lab.io import read_object, write_new, write_state

from raw_training_labs import jobs
from raw_training_labs.cli import main
from raw_training_labs.demo import create, examples, review_synthetic
from raw_training_labs.profiles import decision_messages, supported
from raw_training_labs.scoring import aggregate, comparison, score
from raw_training_labs.server import application
from raw_training_labs.service import Service
from raw_training_labs.storage import Store, new_id
from raw_training_labs.worker import force_stop


@pytest.fixture
def workbench(tmp_path):
    return Service(tmp_path / "raw-private")


def reviewed(service):
    project_id = create(service)["project"]["id"]
    assert review_synthetic(service, project_id)["eligible"]
    return project_id


def test_independent_store_refuses_source_and_foreign_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    with pytest.raises(ValueError, match="outside"):
        Store(source / "private")
    foreign = tmp_path / "existing"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("original")
    with pytest.raises(ValueError, match="adopting"):
        Store(foreign)
    assert (foreign / "keep.txt").read_text() == "original"


def test_review_revisions_preserve_original_and_missing_rights_block(workbench):
    pid = create(workbench)["project"]["id"]
    first = workbench.examples(pid)[0]
    assert first["review"] is None and not workbench.validate(pid)["eligible"]
    rights = dict(first["original"]["rights"], evaluation=False)
    payload = {
        "reviewer": "Test author",
        "reviewer_kind": "delegated_semantic",
        "reviewer_authored_material": True,
        "approved": True,
        "completion": first["original"]["completion"],
        "rights": rights,
        "reason": "Instructional test of explicit permission handling",
    }
    workbench.review(pid, first["id"], payload)
    payload["completion"] = '{"site":null,"trade":null,"urgency":"routine","summary":null}'
    result = workbench.review(pid, first["id"], payload)
    assert result["revision"] == 2
    saved = workbench.examples(pid)[0]
    assert saved["original"] == first["original"]
    assert saved["review"]["completion"] == payload["completion"]
    assert any(p["code"] == "permission_missing" for p in workbench.validate(pid)["problems"])


def test_duplicate_import_is_atomic_and_never_silent(workbench):
    pid = workbench.create_project(
        {"name": "Input check", "customer": "RAW", "task": "Check exact service input import"}
    )["id"]
    rows = examples()
    with pytest.raises(ValueError, match="duplicate"):
        workbench.import_examples(pid, {"examples": [rows[0], rows[0]]})
    assert workbench.examples(pid) == []
    workbench.import_examples(pid, {"examples": rows})
    with pytest.raises(ValueError, match="already imported"):
        workbench.import_examples(pid, {"examples": [rows[0]]})
    assert len(workbench.examples(pid)) == 60


def test_wrong_project_context_and_paths_are_rejected(workbench):
    first, second = reviewed(workbench), reviewed(workbench)
    frozen = workbench.prepare(first)
    with pytest.raises(ValueError, match="this project"):
        workbench.store.artifact(second, frozen["id"], "corpus")
    for name in ("../escape", "C:/escape", "a\\escape", "/absolute"):
        with pytest.raises(ValueError):
            workbench.store.artifact_path(first, name)
    for bad in ("", "missing", "a" * 31):
        with pytest.raises(ValueError):
            workbench.summary(bad)


def test_groups_conflicts_and_chronology_prevent_freeze(workbench):
    pid = workbench.create_project(
        {
            "name": "Conflicts",
            "customer": "RAW",
            "task": "Check overlapping independent split families",
        }
    )["id"]
    rows = examples()
    rows[36]["group"] = rows[0]["group"]
    rows[36]["available_at"] = 1
    rows[37]["prompt"] = copy.deepcopy(rows[0]["prompt"])
    workbench.import_examples(pid, {"examples": rows})
    review_synthetic(workbench, pid)
    codes = {p["code"] for p in workbench.validate(pid)["problems"]}
    assert {"group_crosses_splits", "chronology_overlap", "conflicting_target"} <= codes
    with pytest.raises(ValueError, match="blocked"):
        workbench.prepare(pid)
    assert workbench.artifacts(pid) == []


def test_sealed_targets_absent_from_metadata_and_not_inspected_by_validation(
    workbench, monkeypatch
):
    pid = reviewed(workbench)
    frozen = workbench.prepare(pid)
    artifact = workbench.store.artifact(pid, frozen["id"], "corpus")
    manifest = read_object(artifact["path"] / "manifest.json")
    assert all("completion" not in r["review"] for r in manifest["source_reviews"].values())
    monkeypatch.setattr(workbench, "_examples", lambda *a, **k: pytest.fail("sealed rows read"))
    assert workbench.validate(pid)["eligible"]
    with pytest.raises(ValueError, match="frozen"):
        workbench.prepare(pid)
    with pytest.raises(ValueError, match="frozen"):
        workbench.import_examples(pid, {"examples": [examples()[0]]})


def test_sealed_examples_are_redacted_after_freeze(workbench):
    pid = reviewed(workbench)
    workbench.prepare(pid)
    final = [r for r in workbench.examples(pid) if r["original"]["split"] == "test"]
    assert len(final) == 8
    assert all(r["original"]["completion"] == "[sealed final target]" for r in final)
    assert all(r["review"]["completion"] == "[sealed final target]" for r in final)


def test_global_frozen_exposure_blocks_cross_project_reuse(workbench):
    first = reviewed(workbench)
    workbench.prepare(first)
    second = workbench.create_project(
        {
            "name": "Reuse check",
            "customer": "RAW",
            "task": "Try to reuse a sealed population under new IDs",
        }
    )["id"]
    rows = [r["original"] for r in workbench.examples(first) if r["original"]["split"] != "test"]
    original = examples(first)
    # Full original input population, shifted identities/groups; exact evidence still links it.
    for row in original:
        row["group"] = "changed-" + row["group"]
    workbench.import_examples(second, {"examples": original})
    review_synthetic(workbench, second)
    with pytest.raises(ValueError, match="Protected exposure"):
        workbench.prepare(second)
    assert rows and not workbench.artifacts(second)


def test_tev_has_explicit_decision_template_but_no_inferred_training_support():
    messages = decision_messages(
        "urgent leak",
        "Which team?",
        [
            {"key": "plumbing", "description": "Plumbing"},
            {"key": "electrical", "description": "Electrical"},
        ],
    )
    body = json.loads(messages[1]["content"])
    assert body["options"][0]["label"] == "A" and body["options"][1]["label"] == "B"
    with pytest.raises(ValueError, match="unsupported"):
        supported("tev1-4b-decision")
    with pytest.raises(ValueError, match="2-24"):
        decision_messages("state", "question", [])


@pytest.mark.parametrize("raw", ["{}", "null", "[]", '{"site":null}', "not JSON"])
def test_missing_fields_and_invalid_outputs_never_count_as_success(raw):
    criteria = {"task": "work_request", "fields": ["site", "trade"]}
    outcome = score(raw, '{"site":null,"trade":null}', criteria)
    assert not outcome["success"] and not outcome["valid"]


def test_comparison_counts_failures_regressions_and_resource_metrics():
    criteria = {
        "task": "work_request",
        "fields": ["site"],
        "minimum_success": 0.8,
        "maximum_regressions": 0,
        "maximum_invalid": 0,
        "maximum_p95_seconds": 60.0,
    }

    def case(raw, status):
        return {
            "id": "case",
            "split": "regression",
            "input_sha256": "same",
            "latency_seconds": 1.5,
            "rss_bytes": 1024,
            "output_tokens": 2,
            "status": status,
            "score": score(raw, '{"site":"Unit 1"}', criteria, call_status=status),
        }

    base, candidate = [case('{"site":"Unit 1"}', "ok")], [case("", "failed")]
    outcome = comparison(base, candidate, criteria)
    assert outcome["regressions"] == 1 and not outcome["criteria_passed"]
    assert outcome["candidate"]["cases"] == outcome["candidate"]["call_failures"] == 1
    assert aggregate(candidate)["p95_seconds"] == 1.5


def test_local_requests_require_host_origin_cookie_and_token(tmp_path):
    client = TestClient(application(tmp_path / "server-store"), base_url="http://127.0.0.1:8765")
    assert client.get("/").status_code == 200
    assert client.get("/api/projects").status_code == 403
    token = client.get("/api/session").json()["token"]
    payload = {"name": "HTTP test", "customer": "RAW", "task": "Check local project requests"}
    assert client.post("/api/projects", json=payload).status_code == 403
    headers = {"Origin": "http://127.0.0.1:8765", "X-RAW-CSRF": token}
    good = client.post("/api/projects", json=payload, headers=headers)
    assert good.status_code == 200
    assert client.get("/api/projects").json()[0]["id"] == good.json()["id"]
    assert client.get("/api/projects", headers={"Host": "evil.invalid"}).status_code == 403
    assert (
        client.post(
            "/api/projects", json=payload, headers={**headers, "Origin": "https://evil.invalid"}
        ).status_code
        == 403
    )
    assert (
        client.post("/api/projects", content="x" * (2 * 1024**2 + 1), headers=headers).status_code
        == 413
    )
    assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


def test_cli_and_api_share_projects_and_reliable_failure_codes(workbench, capsys):
    project_id = create(workbench)["project"]["id"]
    assert main(["--root", str(workbench.store.root), "projects"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == project_id
    assert main(["--root", str(workbench.store.root), "summary", "--project", "invalid"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]


def test_interrupted_job_never_invents_success_and_denied_inspection_holds_admission(
    workbench, monkeypatch
):
    pid = create(workbench)["project"]["id"]
    job_id = new_id()
    with workbench.store.transaction() as db:
        db.execute(
            "INSERT INTO jobs(id,project,kind,status,request,key,created) VALUES(?,?,?,?,?,?,?)",
            (job_id, pid, "train", "running", "{}", "interruption-check", time.time() - 60),
        )
    monkeypatch.setattr(jobs, "process_alive", lambda job: None)
    jobs.reconcile(workbench.store)
    assert workbench.store.job(pid, job_id)["status"] == "running"
    monkeypatch.setattr(jobs, "process_alive", lambda job: False)
    jobs.reconcile(workbench.store)
    assert workbench.store.job(pid, job_id)["status"] == "interrupted"
    assert "No clean checkpoint" in workbench.store.job(pid, job_id)["reason"]


def test_exact_job_identity_refuses_replaced_base_and_source(workbench):
    pid = create(workbench)["project"]["id"]
    job_id = new_id()
    directory = workbench.store.artifact_path(pid, f"jobs/{job_id}")
    directory.mkdir(parents=True)
    current = {
        "base_sha256": "base-a",
        "corpus_sha256": "data-a",
        "recipe": {},
        "masking": {},
        "environment": {},
        "profile": {},
    }
    write_new(directory / "identity.json", {"preflight": current, "source": jobs.source_identity()})
    jobs.match_identity(workbench.store, pid, job_id, current)
    with pytest.raises(ValueError, match="differs"):
        jobs.match_identity(workbench.store, pid, job_id, {**current, "base_sha256": "base-b"})


def test_acquisition_requires_explicit_authorization_before_any_remote_call(workbench):
    pid = create(workbench)["project"]["id"]
    with pytest.raises(ValueError, match="authorization"):
        jobs.start(workbench, pid, {"kind": "acquire", "idempotency_key": "unauthorized-check"})
    assert jobs.jobs(workbench.store, pid) == []


def mock_launch(monkeypatch):
    launches = []
    monkeypatch.setattr("llm_lab.model.environment", lambda: {"fixture": "no model execution"})
    monkeypatch.setattr(jobs, "process_alive", lambda job: True)
    monkeypatch.setattr(
        jobs.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 123.0)
    )

    def launch(args, **kwargs):
        launches.append(args)
        return SimpleNamespace(pid=987654)

    monkeypatch.setattr(jobs.subprocess, "Popen", launch)
    return launches


def test_idempotent_job_admission_returns_same_record_without_another_launch(
    workbench, monkeypatch
):
    launches = mock_launch(monkeypatch)
    pid = create(workbench)["project"]["id"]
    request = {"kind": "acquire", "idempotency_key": "repeat-admission", "authorized": True}
    first = jobs.start(workbench, pid, request)
    second = jobs.start(workbench, pid, request)
    assert first["id"] == second["id"] and len(launches) == 1
    with pytest.raises(ValueError, match="different request"):
        jobs.start(workbench, pid, {**request, "yield_after_step": 3})
    assert len(jobs.jobs(workbench.store, pid)) == 1


def test_concurrent_cross_project_admission_launches_only_one_owned_job(workbench, monkeypatch):
    launches = mock_launch(monkeypatch)
    pids = [create(workbench)["project"]["id"] for _ in range(2)]

    def attempt(pid):
        try:
            return jobs.start(
                workbench,
                pid,
                {"kind": "acquire", "idempotency_key": "concurrent-admission", "authorized": True},
            )["id"]
        except ValueError as exc:
            assert "One heavy RAW job" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(attempt, pids))
    assert sum(value is not None for value in ids) == len(launches) == 1
    with workbench.store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_failed_identity_capture_stops_already_launched_worker_before_releasing_admission(
    workbench, monkeypatch
):
    mock_launch(monkeypatch)
    events = []
    process = SimpleNamespace(
        pid=987654,
        poll=lambda: None,
        kill=lambda: events.append("killed"),
        wait=lambda timeout: events.append("waited"),
    )

    def denied_creation():
        raise jobs.psutil.AccessDenied(987654)

    monkeypatch.setattr(
        jobs.psutil,
        "Process",
        lambda pid: SimpleNamespace(create_time=denied_creation, children=lambda recursive: []),
    )
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: process)
    pid = create(workbench)["project"]["id"]
    with pytest.raises(jobs.psutil.AccessDenied):
        jobs.start(
            workbench,
            pid,
            {"kind": "acquire", "idempotency_key": "failed-identity", "authorized": True},
        )
    assert events == ["killed", "waited"]
    assert jobs.jobs(workbench.store, pid)[0]["status"] == "failed"


@pytest.mark.parametrize("failed_report", ["sqlite", "progress"])
def test_watchdog_still_stops_owned_children_and_exits_when_reporting_fails(
    tmp_path, monkeypatch, failed_report
):
    events = []

    @contextmanager
    def unavailable_database():
        raise sqlite3.OperationalError("persistent database failure")
        yield

    class Exited(BaseException):
        pass

    def exit_process(code):
        events.append(("exit", code))
        raise Exited

    child = SimpleNamespace(terminate=lambda: events.append(("terminate", 1)))
    process = SimpleNamespace(children=lambda recursive: [child])
    monkeypatch.setattr("raw_training_labs.worker.psutil.wait_procs", lambda c, timeout: (c, []))
    monkeypatch.setattr("raw_training_labs.worker.os._exit", exit_process)
    if failed_report == "progress":

        def failed_write(*args):
            raise OSError("full output volume")

        monkeypatch.setattr("raw_training_labs.worker.write_state", failed_write)
    with pytest.raises(Exited):
        force_stop(
            SimpleNamespace(transaction=unavailable_database),
            process,
            tmp_path,
            "owned-job",
            "Resource monitor unavailable",
            100,
            code=71,
        )
    assert events == [("terminate", 1), ("exit", 71)]


def test_storage_accounting_tolerates_entry_removed_after_path_guard(workbench, monkeypatch):
    import raw_training_labs.storage as storage

    transient = workbench.store.root / "progress.json.tmp"
    transient.write_text("moving receipt")
    stable = workbench.store.root / "stable.txt"
    stable.write_bytes(b"x" * 4096)
    original_guard = storage.unlinked

    def concurrent_replace(path):
        guarded = original_guard(path)
        if guarded == transient:
            transient.unlink()
        return guarded

    monkeypatch.setattr(storage, "unlinked", concurrent_replace)
    assert workbench.store.budget()["used_bytes"] >= stable.stat().st_size


def test_storage_accounting_has_no_file_type_then_size_race(workbench, monkeypatch):
    transient = workbench.store.root / "progress.json.tmp"
    transient.write_text("moving receipt")
    original_is_file = Path.is_file

    def replaced_after_type_check(path):
        result = original_is_file(path)
        if path == transient and result:
            transient.unlink()
        return result

    monkeypatch.setattr(Path, "is_file", replaced_after_type_check)
    assert workbench.store.budget()["used_bytes"] > 0


def test_storage_accounting_does_not_hide_inspection_permission_failure(workbench, monkeypatch):
    def denied(path):
        raise PermissionError("inspection denied")

    monkeypatch.setattr("raw_training_labs.storage.unlinked", denied)
    with pytest.raises(PermissionError, match="inspection denied"):
        workbench.store.budget()


def test_concurrent_receipt_publishers_use_independent_temporary_files(tmp_path, monkeypatch):
    destination = tmp_path / "progress.json"
    ready = threading.Barrier(2)
    original_replace = Path.replace

    def replace_together(path, target):
        if target == destination:
            ready.wait(timeout=5)
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writes = [pool.submit(write_state, destination, {"writer": n}) for n in (1, 2)]
        for write in writes:
            write.result(timeout=10)
    assert read_object(destination)["writer"] in (1, 2)
    assert list(tmp_path.glob("*.tmp")) == []


def test_watchdog_records_sql_failure_reason_when_progress_publication_fails(
    workbench, monkeypatch
):
    project_id = create(workbench)["project"]["id"]
    job_id = new_id()
    with workbench.store.transaction() as db:
        db.execute(
            "INSERT INTO jobs(id,project,kind,status,request,key,created) VALUES(?,?,?,?,?,?,?)",
            (job_id, project_id, "train", "running", "{}", "forced-stop-report", time.time()),
        )

    class Exited(BaseException):
        pass

    def exit_process(code):
        assert code == 71
        raise Exited

    def failed_write(*args):
        raise OSError("output volume unavailable")

    monkeypatch.setattr("raw_training_labs.worker.write_state", failed_write)
    monkeypatch.setattr("raw_training_labs.worker.os._exit", exit_process)
    monkeypatch.setattr("raw_training_labs.worker.psutil.wait_procs", lambda c, timeout: ([], []))
    with pytest.raises(Exited):
        force_stop(
            workbench.store,
            SimpleNamespace(children=lambda recursive: []),
            workbench.store.root,
            job_id,
            "Resource inspection unavailable",
            100,
            code=71,
        )
    retained = workbench.store.job(project_id, job_id)
    assert retained["status"] == "failed"
    assert retained["reason"] == "Resource inspection unavailable"
