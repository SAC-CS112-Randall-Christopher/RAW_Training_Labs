from llm_lab.io import new_directory, write_new
from llm_lab.report import render
from llm_lab.training import train


def test_report_preserves_proof_limits_and_escapes_untrusted_comparison(hybrid, tmp_path):
    root, recipe = hybrid
    train(root / "base", root / "corpus", recipe, tmp_path / "run")
    comparison = new_directory(tmp_path / "comparison")
    write_new(
        comparison / "comparison.json",
        {
            "summary": {"baseline": {"critical": "<script>bad()</script>"}, "candidate": {}},
        },
    )
    report = render(tmp_path / "run", tmp_path / "report.html", comparison)
    text = report.read_text(encoding="utf-8")
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "random fixtures" in text and "economic improvement" in text
    assert "Skipped AMP attempts" in text and "resource waits" in text
