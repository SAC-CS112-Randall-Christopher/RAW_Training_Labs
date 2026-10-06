"""RAW-authored synthetic service requests. No observed customer or trading records."""

from llm_lab.io import canonical

from raw_training_labs.service import Service

INSTRUCTION = (
    "Convert the service note to JSON with exactly site, trade, urgency, summary. "
    "Copy the stated site and issue into site and summary. trade is plumbing, "
    "electrical, or hvac. urgency is urgent only when explicitly urgent, otherwise "
    "routine. Use null for missing site, trade or issue. Return only JSON."
)
RIGHTS = {
    "training": True,
    "evaluation": True,
    "external_processing": False,
    "sharing": False,
    "export": True,
    "retention": True,
    "basis": "RAW-authored synthetic instructional examples, authorized by Chris Randall; "
    "no customer records.",
}


def examples(cohort=""):
    result = []
    issues = {
        "plumbing": ["tap drips", "sink blocked", "toilet leaks"],
        "electrical": ["lamp flickers", "socket loose", "switch broken"],
        "hvac": ["fan noisy", "filter dirty", "vent blocked"],
    }
    number = 0
    for split, count, start in (
        ("train", 36, 10),
        ("validation", 8, 100),
        ("test", 8, 200),
        ("regression", 8, 300),
    ):
        for index in range(count):
            number += 1
            trade = ["plumbing", "electrical", "hvac"][index % 3]
            issue = issues[trade][(index // 3) % 3]
            site = f"Unit {cohort[:6]}-{start + index}" if cohort else f"Unit {start + index}"
            urgency = "urgent" if index % 4 == 0 else "routine"
            note = f"Site: {site}. Trade: {trade}. Issue: {issue}. Priority: {urgency}."
            target = {"site": site, "trade": trade, "urgency": urgency, "summary": issue}
            if split == "regression" and index % 2 == 0:
                note = f"Trade: {trade}. Issue: {issue}. Priority: {urgency}."
                target["site"] = None
            if split == "regression" and index == 7:
                note = f"Site: {site}. Issue: {issue}. Priority: {urgency}."
                target["trade"] = None
            if cohort:
                note = f"Request: {cohort[:6]}-{split}-{index}. " + note
            result.append(
                {
                    "id": f"raw-service-{split}-{index:02d}",
                    "prompt": [
                        {"role": "system", "content": INSTRUCTION},
                        {"role": "user", "content": note},
                    ],
                    "completion": canonical(target),
                    "group": f"{cohort}service-request-{split}-{index:02d}",
                    "split": split,
                    "available_at": number * 60,
                    "source_kind": "raw_synthetic",
                    "rights": dict(RIGHTS),
                }
            )
    return result


def create(service: Service):
    project = service.create_project(
        {
            "name": "Synthetic service requests",
            "customer": "RAW instructional demo",
            "task": "Extract stated site, trade, urgency and issue from synthetic service notes; "
            "preserve missing facts.",
            "criteria": {
                "task": "work_request",
                "fields": ["site", "trade", "urgency", "summary"],
                "minimum_success": 0.8,
                "maximum_regressions": 0,
                "maximum_invalid": 0,
                "maximum_p95_seconds": 60.0,
                "notes": "Synthetic exact-field extraction; semantic and customer acceptance "
                "remain separate.",
            },
            "approach": "baseline",
        }
    )
    service.import_examples(project["id"], {"examples": examples(project["id"])})
    return {
        "project": project,
        "next_action": "Review original examples before preparation",
        "authorship": "Codex for RAW; synthetic, not empirical",
    }


def review_synthetic(service, project_id):
    """Explicitly invoked synthetic author review; never claims an independent human review."""
    for item in service.examples(project_id):
        original = item["original"]
        if original["source_kind"] != "raw_synthetic" or not original["id"].startswith(
            "raw-service-"
        ):
            raise ValueError(
                "This convenience action is restricted to authored RAW service fixtures"
            )
        service.review(
            project_id,
            item["id"],
            {
                "reviewer": "Codex / RAW synthetic author",
                "reviewer_kind": "delegated_semantic",
                "reviewer_authored_material": True,
                "approved": True,
                "completion": original["completion"],
                "rights": original["rights"],
                "reason": "Author checked literal extraction, explicit missing values, groups "
                "and permissions; instructional scope only.",
            },
        )
    return service.validate(project_id)
