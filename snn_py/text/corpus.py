"""Utility helpers to load small training corpora for the narrator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import gzip
import re

_SENTENCE_SPLIT = re.compile(r"[\.!\?\n]+")
# Match ASCII words/digits (for English logs) or single CJK characters so that
# tokenization keeps Chinese sentences instead of dropping them entirely.
_TOKENIZER = re.compile(r"[a-zA-Z0-9']+|[\u4E00-\u9FFF]")


def _tokenize(sentence: str) -> List[str]:
    tokens = [tok.lower() for tok in _TOKENIZER.findall(sentence)]
    return [tok for tok in tokens if tok]


def _iter_texts(root: Path) -> Iterable[str]:
    if root.is_file():
        if root.suffix == ".gz" or root.name.endswith(".gz"):
            with gzip.open(root, "rt", encoding="utf-8") as handle:
                yield handle.read()
        else:
            yield root.read_text(encoding="utf-8")
        return
    if root.is_dir():
        candidates = list(root.rglob("*.txt")) + list(root.rglob("*.txt.gz"))
        for path in sorted(candidates):
            if path.suffix == ".gz" or path.name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    yield handle.read()
            else:
                yield path.read_text(encoding="utf-8")
        return
    raise FileNotFoundError(root)


def load_corpus(path: Path | str) -> List[List[str]]:
    """Load tokens grouped by sentences from a file or directory of .txt files."""
    root = Path(path)
    sentences: List[List[str]] = []
    for text in _iter_texts(root):
        for chunk in _SENTENCE_SPLIT.split(text):
            chunk = chunk.strip()
            if not chunk:
                continue
            tokens = _tokenize(chunk)
            if tokens:
                sentences.append(tokens)
    return sentences


__all__ = ["load_corpus"]
