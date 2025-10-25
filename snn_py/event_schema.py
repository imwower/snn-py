"""Shared event codes and log validation utilities."""

from __future__ import annotations

from typing import Dict, Literal, Mapping, Union

EventCode = Literal[
    "LOGGING_SETUP",
    "MANIFEST_SAVED",
    "METRICS_DONE",
    "UP_START",
    "DOWN_START",
    "AVALANCHE",
    "INTENT_FIRED",
    "RATE_LIMITED",
    "REFRACTORY",
    "AUDIT_DECISION",
    "EPISODE_APPENDED",
    "PIPELINE_START",
    "PIPELINE_COMPLETE",
    "SWEEP_CASE_DONE",
    "SWEEP_COMPLETE",
    "REPLAY_LOADED",
    "REPLAY_SUMMARY",
    "CHECKPOINT_SAVED",
    "CHECKPOINT_LOADED",
    "RUN_COMPLETE",
    "REPLAY_RECOMPUTED",
    "REPLAY_DIFF",
]

EVENT_CODES: Dict[str, EventCode] = {
    "logging_setup": "LOGGING_SETUP",
    "manifest_saved": "MANIFEST_SAVED",
    "metrics_done": "METRICS_DONE",
    "up_start": "UP_START",
    "上状态开始": "UP_START",
    "down_start": "DOWN_START",
    "下状态开始": "DOWN_START",
    "神经雪崩": "AVALANCHE",
    "intent_fired": "INTENT_FIRED",
    "意图触发": "INTENT_FIRED",
    "速率限制": "RATE_LIMITED",
    "rate_limited": "RATE_LIMITED",
    "不应期": "REFRACTORY",
    "refractory": "REFRACTORY",
    "audit_decision": "AUDIT_DECISION",
    "episode_appended": "EPISODE_APPENDED",
    "pipeline_start": "PIPELINE_START",
    "pipeline_complete": "PIPELINE_COMPLETE",
    "sweep_case_done": "SWEEP_CASE_DONE",
    "sweep_complete": "SWEEP_COMPLETE",
    "replay_loaded": "REPLAY_LOADED",
    "replay_summary": "REPLAY_SUMMARY",
    "replay_recomputed": "REPLAY_RECOMPUTED",
    "replay_diff": "REPLAY_DIFF",
    "checkpoint_saved": "CHECKPOINT_SAVED",
    "checkpoint_loaded": "CHECKPOINT_LOADED",
    "run_complete": "RUN_COMPLETE",
}


def validate_log_dict(payload: Mapping[str, object]) -> Union[bool, str]:
    """Ensure a log dictionary carries the expected schema."""

    if not isinstance(payload, Mapping):
        return "payload must be a mapping"

    event = payload.get("event")
    if not isinstance(event, str):
        return "event missing or not a string"
    if event not in EVENT_CODES:
        return f"unknown event: {event}"

    code = payload.get("code")
    if not isinstance(code, str):
        return "code missing or not a string"

    expected_code = EVENT_CODES[event]
    if code != expected_code:
        return f"code mismatch: expected {expected_code}, got {code}"

    ts = payload.get("ts")
    if not isinstance(ts, (int, float)):
        return "ts missing or not numeric"

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return "meta missing or not a dict"

    return True


__all__ = ["EventCode", "EVENT_CODES", "validate_log_dict"]
