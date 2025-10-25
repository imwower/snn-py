"""Tail helper for append-only JSONL streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import json
import os


class JSONLTailer:
    """Incrementally read newly appended JSONL lines with offset persistence."""

    def __init__(self, path: Path, start_at_end: bool = True):
        self.path = Path(path)
        self.offset = 0
        self._fh: Optional[io.TextIOWrapper] = None
        if self.path.exists():
            self._fh = self.path.open("r", encoding="utf-8")
            if start_at_end:
                self._fh.seek(0, os.SEEK_END)
                self.offset = self._fh.tell()

    def load_state(self, state_path: Path) -> None:
        """Restore tail offset from `state_path` if it matches the same file."""
        state_file = Path(state_path)
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return
        if data.get("file") != str(self.path):
            return
        try:
            self.offset = max(0, int(data.get("offset", 0)))
        except (TypeError, ValueError):
            self.offset = 0
        if self._fh is None and self.path.exists():
            self._fh = self.path.open("r", encoding="utf-8")
        if self._fh is not None:
            try:
                self._fh.seek(self.offset, os.SEEK_SET)
            except OSError:
                self._fh.seek(0, os.SEEK_SET)
                self.offset = 0

    def save_state(self, state_path: Path) -> None:
        """Persist the current offset so the tail can resume later."""
        payload = {"file": str(self.path), "offset": self.offset}
        state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload), encoding="utf-8")

    def close(self) -> None:
        """Close the underlying file handle, if open."""
        if self._fh:
            self._fh.close()
            self._fh = None

    def read_new(self, max_lines: int = 1000) -> List[Dict[str, Any]]:
        """Read up to `max_lines` newly appended JSON objects."""
        if self._fh is None:
            if not self.path.exists():
                return []
            self._fh = self.path.open("r", encoding="utf-8")

        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.offset:
            # File got truncated or rotated.
            self._fh.close()
            self._fh = self.path.open("r", encoding="utf-8")
            self.offset = 0

        collected: List[Dict[str, Any]] = []
        lines_read = 0
        while lines_read < max_lines:
            line = self._fh.readline()
            if not line:
                break
            self.offset = self._fh.tell()
            line = line.strip()
            if not line:
                continue
            try:
                collected.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines and keep going.
                pass
            lines_read += 1
        return collected


__all__ = ["JSONLTailer"]
