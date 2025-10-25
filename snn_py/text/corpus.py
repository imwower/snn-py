"""Utility helpers to load small training corpora for the narrator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import gzip
import re

try:  # pragma: no cover - optional dependency
    import jieba as _jieba  # type: ignore
except Exception:  # pragma: no cover - fallback when jieba missing
    _jieba = None

_SENTENCE_SPLIT = re.compile(r"[\.!\?\n]+")
_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9']+")
_CJK_SPAN = re.compile(r"[\u4E00-\u9FFF]+")
_CJK_CHAR = re.compile(r"[\u4E00-\u9FFF]")
_DEFAULT_TOKENIZER = re.compile(r"[a-zA-Z0-9']+|[\u4E00-\u9FFF]")


def _tokenize(sentence: str) -> List[str]:
    sentence = sentence.strip()
    if not sentence:
        return []

    if _jieba is not None and _CJK_CHAR.search(sentence):
        tokens: List[str] = []
        # jieba preserves semantic chunks; keep ASCII tokens lowercase.
        for tok in _jieba.lcut(sentence, cut_all=False):
            tok = tok.strip()
            if not tok:
                continue
            if _ASCII_TOKEN.fullmatch(tok):
                tokens.append(tok.lower())
            elif _CJK_CHAR.search(tok):
                tokens.append(tok)
        if tokens:
            return tokens

    tokens: List[str] = []
    for tok in _DEFAULT_TOKENIZER.findall(sentence):
        if _CJK_CHAR.search(tok):
            tokens.extend(_chunk_cjk(tok))
        else:
            tokens.append(tok.lower())
    return tokens


def _chunk_cjk(span: str) -> List[str]:
    """Fallback segmentation when jieba is unavailable."""
    if len(span) <= 2:
        return [span]
    chunks: List[str] = []
    step = 2
    for idx in range(0, len(span), step):
        chunk = span[idx : idx + step]
        chunks.append(chunk)
    return chunks


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
