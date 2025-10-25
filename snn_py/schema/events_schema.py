"""Simple schema helpers for event JSONL validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

CURRENT_SCHEMA_VERSION = "1.0"

EVENT_CODES = {
    "LOGGING_SETUP",
    "MANIFEST_SAVED",
    "SEGMENT",
    "PROPOSAL",
    "AUDIT_DECISION",
    "PIPELINE_START",
    "PIPELINE_COMPLETE",
    "REPLAY_SUMMARY",
    "REPLAY_DIFF",
    "REPLAY_RECOMPUTED",
    "REPLAY_MATCH_RATE",
    "METRICS_DONE",
    "ERROR",
}


@dataclass
class ValidationStats:
    total: int = 0
    ok: int = 0
    bad: int = 0
    samples_bad: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "ok": self.ok,
            "bad": self.bad,
            "samples_bad": self.samples_bad[:5],
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ok_event_code(code: Optional[str]) -> bool:
    if code is None:
        return True
    return code in EVENT_CODES


def validate_event(entry: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, "not_dict"

    for key in ("event", "ts"):
        if key not in entry:
            return False, f"missing_{key}"

    if not isinstance(entry["event"], str):
        return False, "event_not_str"
    if not _is_number(entry["ts"]):
        return False, "ts_not_number"

    code = entry.get("code")
    if not _ok_event_code(code):
        return False, "invalid_code"

    event = entry["event"]
    meta = entry.get("meta", {})

    if event == "segment":
        if not isinstance(meta, dict):
            return False, "segment_meta_not_dict"
        q = meta.get("q")
        if q is None or not _is_number(q):
            return False, "segment_missing_q"

    if event == "audit_decision":
        if not isinstance(meta, dict):
            return False, "audit_meta_not_dict"
        if "status" not in meta:
            return False, "audit_missing_status"

    return True, "ok"


def validate_stream(events: List[Dict[str, Any]]) -> ValidationStats:
    stats = ValidationStats()
    for entry in events:
        stats.total += 1
        ok, reason = validate_event(entry)
        if ok:
            stats.ok += 1
        else:
            stats.bad += 1
            if len(stats.samples_bad) < 5:
                stats.samples_bad.append({"reason": reason, "e": entry})
    return stats


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "EVENT_CODES",
    "ValidationStats",
    "validate_event",
    "validate_stream",
]
