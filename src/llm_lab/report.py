"""Offline readable receipt with a measured training curve and escaped source text."""

import html
from pathlib import Path

from llm_lab.corpus import verify_corpus
from llm_lab.io import private_path, read_object


def render(run: Path, destination: Path, comparison: Path | None = None):
    receipt = read_object(run / "run.json")
    measured = read_object(comparison / "comparison.json") if comparison else None
    corpus, _ = verify_corpus(Path(receipt["corpus_directory"]))
    fixture_notice = (
        "Locally constructed random fixtures contain no pretrained Qwen3.5-4B knowledge."
        if corpus.get("source") == "locally-authored-procedural-fixture"
        else "The source application retains semantic review and operating qualification authority."
    )
    points = receipt["history"]
    history = "".join(
        f"<tr><td>{point['step']}</td><td>{point['training_loss']:.6f}</td>"
        f"<td>{html.escape(str(point.get('validation_loss', '—')))}</td></tr>"
        for point in points
    )
    cells = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in {
            "Run status": receipt["status"],
            "Optimization steps": receipt["step"],
            "Selected step": receipt["best_step"],
            "Baseline validation loss": receipt.get("baseline_validation", {}).get("loss"),
            "Selected validation loss": receipt["best_validation_loss"],
            "Trainable parameters": receipt.get("trainable_parameters"),
            "Measured run seconds": round(receipt.get("wall_seconds", 0), 3),
            "Device": receipt["recipe"]["device"],
            "Quantization": receipt["recipe"]["quantization"],
            "Skipped AMP attempts": len(receipt.get("amp_overflow_retries", [])),
            "Corpus": receipt["corpus_sha256"],
            "Base": receipt["base_sha256"],
        }.items()
    )
    metrics = (
        "requested",
        "complete",
        "errors",
        "resource_waits",
        "timeouts",
        "json_valid",
        "action_match",
        "critical",
        "false_rejection",
    )
    headers = "".join(f"<th>{key.replace('_', ' ')}</th>" for key in metrics)
    comparison_text = (
        "<div class='scroll'><table><tr><th>Arm</th>"
        + headers
        + "</tr>"
        + "".join(
            "<tr><td>"
            + arm
            + "</td>"
            + "".join(
                f"<td>{html.escape(str(measured['summary'][arm].get(key, 0)))}</td>"
                for key in metrics
            )
            + "</tr>"
            for arm in measured.get("arms", ["baseline", "candidate"])
        )
        + "</table></div>"
        if measured
        else "<p>Not performed</p>"
    )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Training Lab — recorded run</title>
<style>body{{font:16px system-ui;background:#101820;color:#e9eef3;margin:auto;max-width:950px;
padding:32px}}h1{{font-size:32px}}.notice{{background:#273746;padding:20px;border-radius:12px}}
table{{width:100%;border-collapse:collapse;margin:24px 0}}td,th{{padding:12px;
border-bottom:1px solid #394956;overflow-wrap:anywhere;text-align:left}}
td:first-child{{color:#a9c7db}}
.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>LLM Training Lab</h1><p class="notice">Recorded software experiment. A lower loss or correct
JSON does not establish research quality, operating qualification, or economic improvement.
{fixture_notice}</p>
<table>{cells}</table><details><summary>Measured optimizer history</summary>
<table><tr><th>Step</th><th>Training loss</th><th>Validation loss</th></tr>{history}</table>
</details><p>Checkpoint selection uses validation completion loss. Test data is
reserved for explicit final evaluation.</p><h2>Paired diagnostics</h2>{comparison_text}
<h2>GIS resource policy</h2><p>GPU work defers Monday–Thursday 07:00–18:00 America/Denver.
Yielding occurs at training boundaries; shared WDDM ownership detection has stated limitations.</p>
</html>"""
    destination = private_path(destination)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(body)
    return destination
