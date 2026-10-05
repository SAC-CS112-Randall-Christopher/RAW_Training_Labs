"""Loopback operator UI. All authority and jobs use the same CLI services."""

import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from raw_training_labs import jobs
from raw_training_labs.service import Service


def application(root, port=8765):
    service = Service(root)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    session = secrets.token_urlsafe(32)
    origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    assets = Path(__file__).parent / "web"

    @app.exception_handler(ValueError)
    def invalid(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(ValidationError)
    def invalid_payload(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.headers.get("host") not in hosts:
            return JSONResponse({"error": "Loopback host required"}, status_code=403)
        if request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
            return JSONResponse({"error": "Cross-origin requests refused"}, status_code=403)
        if request.headers.get("origin") and request.headers["origin"] not in origins:
            return JSONResponse({"error": "Untrusted origin"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/session":
            if not secrets.compare_digest(request.cookies.get("raw_session", ""), session):
                return JSONResponse(
                    {"error": "Reload the local workbench to establish a session"}, status_code=403
                )
        if request.method not in {"GET", "HEAD"}:
            if request.headers.get("origin") not in origins or not secrets.compare_digest(
                request.headers.get("x-raw-csrf", ""), session
            ):
                return JSONResponse({"error": "Origin and request token required"}, status_code=403)
            length = request.headers.get("content-length", "")
            if not length.isdigit() or int(length) > 2 * 1024**2:
                return JSONResponse({"error": "Require a body of at most 2 MiB"}, status_code=413)
            body = bytearray()
            async for block in request.stream():
                body.extend(block)
                if len(body) > 2 * 1024**2:
                    return JSONResponse(
                        {"error": "Body limit exceeded; nothing was truncated"}, status_code=413
                    )
            request._body = bytes(body)
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'self'; "
                "script-src 'self'; style-src 'self'; connect-src 'self'; "
                "object-src 'none'; frame-ancestors 'none'",
            }
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"app.js", "app.css"}:
            raise ValueError("Unknown asset")
        return FileResponse(assets / name)

    @app.get("/api/session")
    def establish():
        response = JSONResponse({"token": session, "local_only": True})
        response.set_cookie("raw_session", session, httponly=True, samesite="strict", max_age=86400)
        return response

    @app.get("/api/projects")
    def projects():
        return service.projects()

    @app.post("/api/projects")
    def create_project(payload: dict):
        return service.create_project(payload)

    @app.post("/api/demo")
    def create_demo(payload: dict):
        if payload != {"create": True}:
            raise ValueError("Explicit synthetic project creation required")
        from raw_training_labs.demo import create

        return create(service)

    @app.get("/api/profiles")
    def profiles():
        return service.profiles()

    @app.get("/api/projects/{project_id}")
    def summary(project_id: str):
        return service.summary(project_id)

    @app.get("/api/projects/{project_id}/examples")
    def examples(project_id: str):
        return service.examples(project_id)

    @app.post("/api/projects/{project_id}/examples")
    def import_examples(project_id: str, payload: dict):
        return service.import_examples(project_id, payload)

    @app.post("/api/projects/{project_id}/examples/{example_id}/review")
    def review(project_id: str, example_id: str, payload: dict):
        return service.review(project_id, example_id, payload)

    @app.post("/api/projects/{project_id}/demo-review")
    def synthetic_review(project_id: str, payload: dict):
        if payload != {"review": True}:
            raise ValueError("Explicit synthetic author review required")
        from raw_training_labs.demo import review_synthetic

        return review_synthetic(service, project_id)

    @app.get("/api/projects/{project_id}/validation")
    def validation(project_id: str):
        return service.validate(project_id)

    @app.post("/api/projects/{project_id}/prepare")
    def prepare(project_id: str, payload: dict):
        if payload != {"freeze": True}:
            raise ValueError("Explicit data freeze required")
        return service.prepare(project_id)

    @app.post("/api/projects/{project_id}/acquire")
    def acquire(project_id: str, payload: dict):
        if (
            set(payload) != {"authorized", "profile_id", "idempotency_key"}
            or payload["authorized"] is not True
            or payload["profile_id"] != "qwen3-0.6b-fast-cpu"
        ):
            raise ValueError("Only the explicitly authorized pinned demo acquisition is supported")
        return jobs.start(service, project_id, {"kind": "acquire", **payload})

    @app.get("/api/projects/{project_id}/preflight/{corpus_id}/{profile_id}")
    def preflight(project_id: str, corpus_id: str, profile_id: str):
        return service.preflight(project_id, corpus_id, profile_id)

    @app.post("/api/projects/{project_id}/jobs")
    def start(project_id: str, payload: dict):
        return jobs.start(service, project_id, payload)

    @app.get("/api/projects/{project_id}/jobs/{job_id}")
    def job(project_id: str, job_id: str):
        jobs.reconcile(service.store)
        return jobs.detail(service.store, project_id, job_id)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/{action}")
    def control(project_id: str, job_id: str, action: str, payload: dict):
        if payload != {"request": True} or action not in {"cancel", "yield"}:
            raise ValueError("Unsupported recovery action")
        return (jobs.cancel if action == "cancel" else jobs.yield_job)(
            service.store, project_id, job_id
        )

    @app.get("/api/projects/{project_id}/jobs/{job_id}/log")
    def log(project_id: str, job_id: str):
        service.store.job(project_id, job_id)
        path = service.store.artifact_path(project_id, f"jobs/{job_id}/worker.log")
        if not path.is_file():
            return {"text": "No worker log yet"}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            text = stream.read(16000).decode("utf-8", errors="replace")
        return {"text": text}

    @app.get("/api/projects/{project_id}/jobs/{job_id}/assessments")
    def assessments(project_id: str, job_id: str):
        return service.assessments(project_id, job_id)

    @app.post("/api/projects/{project_id}/jobs/{job_id}/assessments")
    def assessment(project_id: str, job_id: str, payload: dict):
        return service.assess(project_id, job_id, payload)

    return app
