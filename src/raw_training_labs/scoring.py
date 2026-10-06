"""Strict task outcomes. Missing keys never earn credit by matching absent values."""

import json
import math


def score(raw, expected, criteria, *, call_status="ok"):
    result = {
        "success": False,
        "valid": False,
        "field_matches": 0,
        "field_total": len(criteria["fields"]),
        "unsupported_claims": 0,
        "limitations": "Exact authored-target scoring; semantic review remains separate.",
    }
    if call_status != "ok":
        return result | {"error": call_status}
    if criteria["task"] in {"exact_text", "decision"}:
        valid = bool(raw.strip())
        if criteria["task"] == "decision":
            valid = len(raw.strip()) == 1 and "A" <= raw.strip() <= "X"
        success = valid and raw.strip() == expected.strip()
        return result | {
            "valid": valid,
            "success": success,
            "field_total": 1,
            "field_matches": int(success),
        }
    try:
        predicted, target = json.loads(raw), json.loads(expected)
        fields = criteria["fields"]
        if (
            not isinstance(predicted, dict)
            or not isinstance(target, dict)
            or set(target) != set(fields)
        ):
            raise ValueError("Require object output and complete declared target fields")
        matches = sum(
            f in predicted and type(predicted[f]) is type(target[f]) and predicted[f] == target[f]
            for f in fields
        )
        valid = set(predicted) == set(fields) and all(
            v is None or isinstance(v, str) for v in predicted.values()
        )
        claims = len(set(predicted) - set(fields)) + sum(
            f in predicted and predicted[f] is not None and predicted[f] != target[f]
            for f in fields
        )
        return result | {
            "valid": valid,
            "success": valid and matches == len(fields),
            "field_matches": matches,
            "unsupported_claims": claims,
        }
    except (ValueError, TypeError) as exc:
        return result | {"error": str(exc)}


def aggregate(cases):
    latencies = sorted(c["latency_seconds"] for c in cases)
    count = len(cases)
    fields = sum(c["score"]["field_total"] for c in cases)
    return {
        "cases": count,
        "successes": sum(c["score"]["success"] for c in cases),
        "success_rate": sum(c["score"]["success"] for c in cases) / count if count else None,
        "invalid_outputs": sum(not c["score"]["valid"] for c in cases),
        "call_failures": sum(c["status"] != "ok" for c in cases),
        "unsupported_claims": sum(c["score"]["unsupported_claims"] for c in cases),
        "field_accuracy": sum(c["score"]["field_matches"] for c in cases) / fields
        if fields
        else None,
        "p95_seconds": latencies[max(0, math.ceil(count * 0.95) - 1)] if count else None,
        "peak_observed_rss_bytes": max((c["rss_bytes"] for c in cases), default=0),
        "output_tokens": sum(c["output_tokens"] for c in cases),
    }


def comparison(baseline, candidate, criteria):
    if {(c["id"], c["split"], c["input_sha256"]) for c in baseline} != {
        (c["id"], c["split"], c["input_sha256"]) for c in candidate
    } or len(baseline) != len(candidate):
        raise ValueError("Both arms must retain every identical evaluation input")
    by_id = {c["id"]: c for c in baseline}
    pairs = [
        {
            "id": c["id"],
            "split": c["split"],
            "baseline": by_id[c["id"]],
            "candidate": c,
            "regression": by_id[c["id"]]["score"]["success"] and not c["score"]["success"],
            "improvement": c["score"]["success"] and not by_id[c["id"]]["score"]["success"],
        }
        for c in candidate
    ]
    first, second = aggregate(baseline), aggregate(candidate)
    regressions = sum(p["regression"] for p in pairs)
    passes = bool(
        second["cases"]
        and second["success_rate"] >= criteria["minimum_success"]
        and regressions <= criteria["maximum_regressions"]
        and second["invalid_outputs"] <= criteria["maximum_invalid"]
        and second["p95_seconds"] <= criteria["maximum_p95_seconds"]
        and second["call_failures"] == 0
    )
    return {
        "baseline": first,
        "candidate": second,
        "regressions": regressions,
        "improvements": sum(p["improvement"] for p in pairs),
        "criteria_passed": passes,
        "task_improvement": second["successes"] > first["successes"] and passes,
        "pairs": pairs,
        "next_action": "Review disagreements and customer suitability"
        if passes
        else "Keep candidate in development; inspect failures before a fresh "
        "independently held-out iteration",
        "limitations": [
            "Exact-target scoring and synthetic inputs do not establish customer acceptance.",
            "Resource measurements belong to this CPU profile; output length is diagnostic only.",
        ],
    }
