"""策略审计器，基于 JSON 策略文件约束操作提案。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from snn_py import logging_config


def _get_logger() -> logging.Logger:
    return logging_config.get_logger("snn_py.policy.auditor")


@dataclass(frozen=True)
class AuditResult:
    """审计结果，描述决策状态与理由。"""

    status: str
    reason: str


class Auditor:
    """根据策略文件审计代理的工具调用。"""

    def __init__(self, policy_path: str) -> None:
        try:
            with open(policy_path, "r", encoding="utf-8") as fh:
                policy = json.load(fh)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"策略文件不存在: {policy_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"策略文件 JSON 解析失败: {policy_path}") from exc

        tools = policy.get("tools", {})
        self._allowlist = set(tools.get("allowlist", []))
        self._guarded = set(tools.get("guarded", []))
        self._rules: Dict[str, Dict[str, Any]] = tools.get("rules", {})

    def check(self, tool: str, args: Dict[str, Any]) -> AuditResult:
        """审计给定工具提案，返回决策信息。"""

        status: str
        reason: str

        if tool in self._allowlist:
            status = "APPROVED"
            reason = "工具在允许列表"
        elif tool in self._guarded:
            rule = self._rules.get(tool, {})
            modes = rule.get("modes", [])
            require_human = bool(rule.get("require_human_confirm"))
            mode = args.get("mode")
            if mode in modes and require_human:
                status = "REQUIRES_HUMAN"
                reason = "需要人工确认"
            else:
                status = "DENIED"
                reason = "未满足受控工具条件"
        else:
            status = "DENIED"
            reason = "工具未被授权"

        logger = _get_logger()
        logger.info(
            json.dumps(
                {"event": "审计决策", "meta": {"tool": tool, "status": status}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        return AuditResult(status=status, reason=reason)
