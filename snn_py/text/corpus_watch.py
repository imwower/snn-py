"""Detect changes in corpus directories for incremental training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

Fingerprint = Dict[str, Tuple[int, float]]


def dir_fingerprint(directory: Path) -> Fingerprint:
    """Capture file size + mtime for *.txt / *.txt.gz files under `directory`."""
    directory = Path(directory)
    snapshot: Fingerprint = {}
    if not directory.exists():
        return snapshot
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".txt" or path.name.endswith(".txt.gz"):
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path.resolve())] = (stat.st_size, stat.st_mtime)
    return snapshot


def changed(previous: Fingerprint, current: Fingerprint) -> bool:
    """Return True when fingerprints differ."""
    return previous != current


def load_fp(state_path: Path) -> Fingerprint:
    """Load a fingerprint snapshot from disk, if available."""
    state_path = Path(state_path)
    if not state_path.exists():
        return {}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    snapshot: Fingerprint = {}
    for key, value in raw.items():
        try:
            size = int(value[0])
            mtime = float(value[1])
        except (TypeError, ValueError, IndexError):
            continue
        snapshot[str(key)] = (size, mtime)
    return snapshot


def save_fp(state_path: Path, fingerprint: Fingerprint) -> None:
    """Persist fingerprint snapshot for later comparisons."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {path: [size, mtime] for path, (size, mtime) in fingerprint.items()}
    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


__all__ = ["dir_fingerprint", "changed", "load_fp", "save_fp"]

