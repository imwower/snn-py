"""策略审计器：允许、受控、拒绝，并包含策略校验。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from snn_py import logging_config


def _get_logger() -> logging.Logger:
    return logging_config.get_logger("snn_py.policy.auditor")


@dataclass(frozen=True)
class AuditResult:
    status: str
    reason: str


class Auditor:
    """读取策略文件并审计工具调用。"""

    def __init__(self, policy_path: str) -> None:
        try:
            with open(policy_path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"策略文件不存在: {policy_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"策略文件 JSON 解析失败: {policy_path}") from exc

        ok, reason = self._validate(doc)
        if not ok:
            raise ValueError(f"策略文件结构不合法: {reason}")

        tools = doc["tools"]
        self._allowlist = set(tools.get("allowlist", []))
        self._guarded = set(tools.get("guarded", []))
        self._rules: Dict[str, Dict[str, Any]] = tools.get("rules", {})

        logger = _get_logger()
        logger.info(
            json.dumps(
                {"event": "audit_policy_loaded", "meta": {"allow": len(self._allowlist), "guard": len(self._guarded)}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _validate(self, doc: Mapping[str, Any]) -> Tuple[bool, str]:
        if not isinstance(doc, Mapping):
            return False, "策略需为字典"
        tools = doc.get("tools")
        if not isinstance(tools, Mapping):
            return False, "缺少 tools 或类型错误"
        allow = tools.get("allowlist")
        guard = tools.get("guarded")
        rules = tools.get("rules")
        if not (isinstance(allow, list) and all(isinstance(x, str) for x in allow)):
            return False, "allowlist 必须是字符串列表"
        if not (isinstance(guard, list) and all(isinstance(x, str) for x in guard)):
            return False, "guarded 必须是字符串列表"
        if not isinstance(rules, Mapping):
            return False, "rules 必须是字典"
        for key, rule in rules.items():
            if not isinstance(rule, Mapping):
                return False, f"规则 {key} 必须为字典"
            modes = rule.get("modes")
            if not (isinstance(modes, list) and all(isinstance(m, str) for m in modes)):
                return False, f"规则 {key} 的 modes 必须为字符串列表"
            if "require_human_confirm" not in rule:
                return False, f"规则 {key} 缺少 require_human_confirm"
        return True, ""

    def check(self, tool: str, args: Mapping[str, Any]) -> AuditResult:
        logger = _get_logger()
        if tool in self._allowlist:
            status, reason = "APPROVED", "工具在允许列表"
        elif tool in self._guarded:
            rule = self._rules.get(tool, {})
            modes = rule.get("modes", [])
            require_human = bool(rule.get("require_human_confirm"))
            mode = args.get("mode")
            if mode in modes and require_human:
                status, reason = "REQUIRES_HUMAN", "需要人工确认"
            else:
                status, reason = "DENIED", "受控工具条件不满足"
        else:
            status, reason = "DENIED", "工具未被授权"

        logger.info(
            json.dumps(
                {"event": "audit_decision", "meta": {"tool": tool, "status": status}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return AuditResult(status=status, reason=reason)

