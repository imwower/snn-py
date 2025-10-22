"""情景记忆的环形缓冲。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterator, List

from snn_py import logging_config

_LOGGER = logging_config.get_logger("snn_py.memory.episodic")


@dataclass
class Episode:
    """呈现给智能体的情景片段快照。"""

    t0: float
    t1: float
    kind: str
    meta: Dict[str, object]
    payload: Dict[str, object]


class EpisodicStore:
    """支持回放的情景记忆环形缓冲区。"""

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("容量必须为正整数")
        self._capacity = capacity
        self._buffer: List[Episode | None] = [None] * capacity
        self._head = 0
        self._size = 0

    def append(self, ep: Episode) -> None:
        """Insert an episode, overwriting the oldest when full."""
        self._buffer[self._head] = ep
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
        payload = {
            "event": "片段写入",
            "meta": {"kind": ep.kind, "t0": ep.t0, "t1": ep.t1},
        }
        _LOGGER.info(json.dumps(payload, separators=(",", ":")))

    def query_last(self, k: int = 10) -> List[Episode]:
        """返回最新的最多 k 条片段（按时间逆序）。"""
        if k <= 0 or self._size == 0:
            return []
        result: List[Episode] = []
        idx = (self._head - 1) % self._capacity
        for _ in range(min(k, self._size)):
            item = self._buffer[idx]
            assert item is not None
            result.append(item)
            idx = (idx - 1) % self._capacity
        return result

    def replay_iter(self, k: int = 50) -> Iterator[Episode]:
        """按时间逆序迭代最近的最多 k 条片段。"""
        for episode in self.query_last(k):
            yield episode
