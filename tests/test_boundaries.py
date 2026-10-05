from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_lab.config import Recipe
from llm_lab.io import new_directory, read_rows, write_new
from llm_lab.resources import ZONE, gpu_gate, schedule


@pytest.mark.parametrize(
    "at, admitted",
    [
        ("2026-10-05T06:59:59", True),
        ("2026-10-05T07:00:00", False),
        ("2026-10-08T17:59:59", False),
        ("2026-10-08T18:00:00", True),
        ("2026-10-09T12:00:00", True),
        ("2026-10-10T12:00:00", True),
        ("2026-11-02T07:00:00", False),
        ("2027-03-15T07:00:00", False),
    ],
)
def test_gis_schedule_boundaries_including_dst(at, admitted):
    state = schedule(datetime.fromisoformat(at).replace(tzinfo=ZONE))
    assert state["gpu_admitted"] is admitted
    assert state["next_priority_start"] > state["observed_at"]


def test_schedule_converts_utc_without_fixed_offset():
    assert schedule(datetime.fromisoformat("2026-10-05T13:00:00+00:00"))["gpu_admitted"] is False
    assert schedule(datetime.fromisoformat("2026-11-02T13:00:00+00:00"))["gpu_admitted"] is True


def test_protected_hours_do_not_even_inspect_gpu(monkeypatch):
    monkeypatch.setattr("llm_lab.resources._query", lambda *_: pytest.fail("GPU inspection"))
    assert not gpu_gate(now=datetime(2026, 10, 5, 9, tzinfo=ZONE))["gpu_admitted"]


@pytest.mark.parametrize(
    "name, memory, allowed",
    [
        ("chrome.exe", "N/A", True),
        ("ArcGISPro.exe", "N/A", False),
        ("python.exe", "N/A", False),
        ("ollama_llama_server.exe", "N/A", False),
        ("unknown.exe", "500", False),
    ],
)
def test_wddm_shared_gpu_ownership(monkeypatch, name, memory, allowed):
    monkeypatch.setattr("psutil.Process", lambda *_: SimpleNamespace(name=lambda: name))
    monkeypatch.setattr(
        "llm_lab.resources._query",
        lambda fields, kind="gpu": (
            [["42", memory]] if kind == "compute-apps" else [["T1000", "8192", "7000", "597.06"]]
        ),
    )
    assert gpu_gate(now=datetime(2026, 10, 2, 10, tzinfo=ZONE))["gpu_admitted"] is allowed


def test_missing_gpu_inspection_defers(monkeypatch):
    def missing(*_):
        raise OSError("No driver")

    monkeypatch.setattr("llm_lab.resources._query", missing)
    assert not gpu_gate(now=datetime(2026, 10, 2, 10, tzinfo=ZONE))["gpu_admitted"]


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cpu", "precision": "float16"},
        {"device": "cuda", "quantization": "nf4", "precision": "float32"},
        {"warmup_steps": 100},
        {"steps": True},
        {"rank": "16"},
        {"learning_rate": float("nan")},
        {"made_up_option": True},
        {"targets": ["all-linear", "q_proj"]},
    ],
)
def test_invalid_recipe_is_refused(changes):
    with pytest.raises(ValidationError):
        Recipe(name="test", **changes)


def test_git_data_refused_and_existing_artifacts_not_overwritten(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        new_directory(repository / "private")
    owned = new_directory(tmp_path / "owned")
    write_new(owned / "record.json", {"id": "first"})
    with pytest.raises(FileExistsError):
        new_directory(owned)
    with pytest.raises(FileExistsError):
        write_new(owned / "record.json", {"id": "second"})


def test_overlong_jsonl_is_refused_without_truncation(tmp_path):
    file = tmp_path / "large.jsonl"
    file.write_text('{"x":"' + "a" * 262144 + '"}\n')
    with pytest.raises(ValueError, match="limit"):
        read_rows(file)
