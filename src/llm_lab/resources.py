"""Cooperative GIS priority. Never terminates or unloads another process."""

import csv
import os
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/Denver")


def schedule(now: datetime | None = None) -> dict:
    local = (now or datetime.now(ZONE)).astimezone(ZONE)
    protected = local.weekday() < 4 and 7 <= local.hour < 18
    candidates = []
    for offset in range(8):
        day = local.date() + timedelta(days=offset)
        if day.weekday() < 4:
            boundary = datetime(day.year, day.month, day.day, 7, tzinfo=ZONE)
            if boundary > local:
                candidates.append(boundary)
    return {
        "timezone": "America/Denver",
        "gpu_admitted": not protected,
        "reason": "GIS priority hours" if protected else "Outside scheduled GIS priority hours",
        "observed_at": local.isoformat(),
        "next_priority_start": min(candidates).isoformat(),
        "policy": "Defer GPU Monday-Thursday 07:00-18:00; yield to other compute processes",
    }


def _query(fields: str, kind="gpu") -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise ValueError("GPU inspection unavailable; GPU work is deferred")
    return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))


def gpu_gate(minimum_free_mib=1024, *, check_memory=True, now=None) -> dict:
    result = schedule(now)
    if not result["gpu_admitted"]:
        return result
    try:
        import psutil

        others = []
        ambiguous = 0
        for row in _query("pid,used_gpu_memory", "compute-apps"):
            if not row or not row[0].isdigit() or int(row[0]) == os.getpid():
                continue
            if len(row) > 1 and row[1].isdigit():
                if int(row[1]) > 0:
                    others.append(row[0])
                continue
            # WDDM lists ordinary desktop graphics as compute-apps with N/A memory.
            # Treat resident GIS/model/python workers as protected; do not claim
            # exclusive GPU ownership from this lossy driver view.
            ambiguous += 1
            try:
                name = psutil.Process(int(row[0])).name().casefold()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                name = ""
            if name.startswith(("arcgis", "arcsoc", "ollama", "llama")) or name in {
                "python.exe",
                "pythonw.exe",
                "python",
                "python3",
            }:
                others.append(row[0])
        result["ambiguous_graphics_processes"] = ambiguous
        result["ownership_basis"] = "Driver memory plus known GIS/model workers; WDDM is incomplete"
        if others:
            return result | {
                "gpu_admitted": False,
                "reason": "Another GPU compute process owns work",
            }
        gpu = _query("name,memory.total,memory.free,driver_version")[0]
        result.update(
            gpu_name=gpu[0], total_mib=int(gpu[1]), free_mib=int(gpu[2]), driver_version=gpu[3]
        )
        if check_memory and result["free_mib"] < minimum_free_mib:
            result.update(gpu_admitted=False, reason="Insufficient measured free GPU memory")
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        result.update(gpu_admitted=False, reason="GPU ownership or capacity is unavailable")
    return result


def low_priority() -> None:
    import psutil

    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
