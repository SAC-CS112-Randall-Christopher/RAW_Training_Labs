"""Bounded, hash-verified private artifacts. No pickle input or remote fetching."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

MAX_FILE = 64 * 1024 * 1024
MAX_ROW = 262_144
MAX_ROWS = 10_000


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def qtrades_digest(value: Any) -> str:
    # Q-Trades fingerprint deliberately includes json.dumps' default spaces.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Use regular artifact files")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read_object(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError("Expected a bounded regular JSON file")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    canonical(result)
    return result


def read_rows(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError("Expected a bounded regular JSONL file")
    result = []
    with path.open("rb") as stream:
        while line := stream.readline(MAX_ROW + 1):
            if len(line) > MAX_ROW or len(result) >= MAX_ROWS:
                raise ValueError("Row or count limit exceeded; nothing was truncated")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Every row must be a JSON object")
            canonical(row)
            result.append(row)
    return result


def private_path(path: Path) -> Path:
    result = path.resolve()
    if any((parent / ".git").exists() for parent in (result, *result.parents)):
        raise ValueError("Keep datasets, model weights and runs outside Git repositories")
    return result


def new_directory(path: Path) -> Path:
    result = private_path(path)
    result.mkdir(parents=False, exist_ok=False)
    return result


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


def write_state(path: Path, value: Any) -> None:
    # Only an owned mutable receipt; frozen corpus/model manifests use write_new.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(canonical(value) + "\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
