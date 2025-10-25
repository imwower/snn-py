"""情景记忆环形缓冲与 JSONL 持久化。"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional
from uuid import uuid4

from snn_py import logging_config


def _get_logger() -> logging.Logger:
    return logging_config.get_logger("snn_py.memory.episodic")


@dataclass(frozen=True)
class Episode:
    t0: float
    t1: float
    kind: str
    meta: Dict[str, object]
    payload: Dict[str, object]


class EpisodicStore:
    def __init__(
        self,
        capacity: int = 512,
        jsonl_dir: Optional[Path] = None,
        roll_bytes: int = 8_000_000,
        compress: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: List[Optional[Episode]] = [None] * capacity
        self._head = 0
        self._size = 0
        self._logger = _get_logger()

        self._jsonl_dir = Path(jsonl_dir) if jsonl_dir is not None else None
        self._roll_bytes = roll_bytes
        self._compress = compress
        self._run_id = uuid4().hex
        self._part_index = 0
        self._current_file: Optional[Path] = None
        self._fh: Optional[BinaryIO] = None
        self._written_bytes = 0

        if self._jsonl_dir is not None:
            self._jsonl_dir.mkdir(parents=True, exist_ok=True)
            self._open_new_file()

    def _next_file_path(self) -> Path:
        assert self._jsonl_dir is not None
        suffix = ".jsonl.gz" if self._compress else ".jsonl"
        return self._jsonl_dir / f"{self._run_id}.part{self._part_index}{suffix}"

    def _open_new_file(self, rotated: bool = False) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self._next_file_path()
        mode = "ab"
        if self._compress:
            self._fh = gzip.open(path, mode)
        else:
            self._fh = open(path, mode)
        self._current_file = path
        self._written_bytes = 0
        self._part_index += 1
        if rotated:
            self._logger.info("", extra={"event": "jsonl_rotated", "meta": {"next": str(path)}})

    def append(self, episode: Episode) -> None:
        self._buffer[self._head] = episode
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

        offset = None
        filename = None
        if self._fh is not None:
            payload = asdict(episode)
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            data = line.encode("utf-8")
            if self._written_bytes + len(data) > self._roll_bytes:
                self._open_new_file(rotated=True)
            offset = self._written_bytes
            assert self._fh is not None
            self._fh.write(data)
            self._fh.flush()
            self._written_bytes += len(data)
            filename = str(self._current_file)

        self._logger.info(
            "",
            extra={
                "event": "episode_appended",
                "meta": {"kind": episode.kind, "offset": offset, "file": filename},
            },
        )

    def query_last(self, k: int = 10) -> List[Episode]:
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
        for episode in self.query_last(k):
            yield episode

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


__all__ = ["Episode", "EpisodicStore"]
