"""One owned private store, safe artifact paths, transactional project/job metadata."""

import json
import os
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from llm_lab.io import canonical, read_object, write_new

DEFAULT_ROOT = Path("G:/Projects/RAW-Training-Labs-Data")
MAX_BYTES = 20 * 1024**3


def new_id():
    return uuid.uuid4().hex


def identifier(value):
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Use the complete RAW identifier")
    return value


def unlinked(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or (
            item.exists()
            and os.name == "nt"
            and item.stat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("Linked or reparse artifact paths are refused")
    return path.resolve()


class Store:
    def __init__(self, root=DEFAULT_ROOT):
        root = unlinked(root)
        if any((p / ".git").exists() for p in (root, *root.parents)):
            raise ValueError("Private storage must be outside every source checkout")
        if root.exists() and not (root / "raw-store.json").is_file():
            if any(root.iterdir()):
                raise ValueError("Refuse adopting a nonempty directory without RAW ownership")
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "raw-store.json"
        if not marker.exists():
            try:
                write_new(
                    marker,
                    {
                        "format": "raw-training-labs-store-v1",
                        "id": new_id(),
                        "maximum_bytes": MAX_BYTES,
                    },
                )
            except FileExistsError:
                pass
        if read_object(marker).get("format") != "raw-training-labs-store-v1":
            raise ValueError("Unrecognized private store")
        self.root = root
        self.path = root / "metadata.sqlite3"
        unlinked(self.path)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects(
                  id TEXT PRIMARY KEY, body TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS examples(
                  project TEXT NOT NULL, id TEXT NOT NULL, original TEXT NOT NULL,
                  PRIMARY KEY(project,id));
                CREATE TABLE IF NOT EXISTS reviews(
                  project TEXT NOT NULL, example TEXT NOT NULL, revision INTEGER NOT NULL,
                  body TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(project,example,revision));
                CREATE TABLE IF NOT EXISTS artifacts(
                  id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
                  relative TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                  id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
                  status TEXT NOT NULL, request TEXT NOT NULL, key TEXT NOT NULL,
                  pid INTEGER, process_created REAL, heartbeat REAL, created REAL NOT NULL,
                  reason TEXT, UNIQUE(project,key));
                CREATE TABLE IF NOT EXISTS assessments(
                  id TEXT PRIMARY KEY, project TEXT NOT NULL, job TEXT NOT NULL,
                  body TEXT NOT NULL, created REAL NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        return db

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def project(self, project_id, db=None):
        identifier(project_id)
        if db is None:
            with self.connect() as connection:
                return self.project(project_id, connection)
        row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise ValueError("Project not found")
        return {"id": row["id"], **json.loads(row["body"]), "created_at": row["created"]}

    def directory(self, project_id):
        self.project(project_id)
        directory = unlinked(self.root / "projects" / identifier(project_id))
        directory.mkdir(parents=True, exist_ok=True)
        return unlinked(directory)

    def artifact_path(self, project_id, relative):
        root = self.directory(project_id)
        if not isinstance(relative, str) or ":" in relative or "\\" in relative:
            raise ValueError("Unsafe artifact identity")
        path = unlinked(root / relative)
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(root)
        ):
            raise ValueError("Artifact is outside its project")
        return path

    def artifact(self, project_id, artifact_id, kind=None):
        self.project(project_id)
        identifier(artifact_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM artifacts WHERE project=? AND id=?", (project_id, artifact_id)
            ).fetchone()
        if row is None or (kind and row["kind"] != kind):
            raise ValueError("Artifact not found in this project or wrong artifact kind")
        return {
            **dict(row),
            "body": json.loads(row["body"]),
            "path": self.artifact_path(project_id, row["relative"]),
        }

    def job(self, project_id, job_id):
        self.project(project_id)
        identifier(job_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE project=? AND id=?", (project_id, job_id)
            ).fetchone()
        if row is None:
            raise ValueError("Job not found in this project")
        return {**dict(row), "request": json.loads(row["request"])}

    def budget(self, reserve=0):
        used = 0
        count = 0
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            for name in dirs + files:
                count += 1
                if count > 100000:
                    raise ValueError("Storage inspection exceeds the bounded file count")
                try:
                    item = unlinked(Path(directory) / name)
                    metadata = item.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # Mutable receipt/download entries can be replaced during inspection.
                    continue
                if not item.is_relative_to(self.root) or stat.S_ISLNK(metadata.st_mode) or (
                    os.name == "nt"
                    and metadata.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise ValueError("Linked or reparse artifact paths are refused")
                if stat.S_ISREG(metadata.st_mode):
                    used += metadata.st_size
            if used + reserve > MAX_BYTES:
                raise ValueError("Approved 20 GiB private-storage budget exceeded")
        return {"used_bytes": used, "maximum_bytes": MAX_BYTES, "reserve_bytes": reserve}

    def add_artifact(self, project_id, artifact_id, kind, relative, body, db=None):
        import time

        self.artifact_path(project_id, relative)
        if db is None:
            with self.transaction() as connection:
                return self.add_artifact(project_id, artifact_id, kind, relative, body, connection)
        db.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?,?,?)",
            (artifact_id, project_id, kind, relative, canonical(body), time.time()),
        )
