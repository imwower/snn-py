"""JSONL helpers for Replay++ and other tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Any
import json
import gzip


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """逐行读取 .jsonl 或 .jsonl.gz"""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def write_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    """追加写入一行 JSON；若后缀为 .gz 则自动 gzip 追加"""
    s = json.dumps(rec, ensure_ascii=False)
    if str(path).endswith(".gz"):
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write(s + "\n")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")


def scan_jsonl_dir(dir_path: Path) -> Iterator[Path]:
    """遍历目录下所有 .jsonl / .jsonl.gz"""
    for p in sorted(dir_path.iterdir()):
        if p.suffix == ".jsonl" or str(p).endswith(".jsonl.gz"):
            yield p


__all__ = ["iter_jsonl", "write_jsonl", "scan_jsonl_dir"]
