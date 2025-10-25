"""Evaluate n-gram model perplexity and OOV coverage on a corpus."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

try:
    from snn_py import logging_config as _logging_config
except Exception:  # pragma: no cover - fallback
    _logging_config = None  # type: ignore[assignment]

from snn_py.text.corpus import load_corpus
from snn_py.text.ngram import NGramModel

_BASIC_LOG_FORMAT = "%(message)s"


def _ensure_logging() -> None:
    if _logging_config is not None:
        try:
            _logging_config.setup()
            return
        except Exception:  # pragma: no cover - fallback
            pass
    logging.basicConfig(level=logging.INFO, format=_BASIC_LOG_FORMAT)


def _get_logger(name: str) -> logging.Logger:
    _ensure_logging()
    if _logging_config is not None:
        try:
            return _logging_config.get_logger(name)
        except Exception:  # pragma: no cover - fallback
            pass
    return logging.getLogger(name)


log = _get_logger("snn_py.cli.text_eval")


def _oov_ratio(sentences, vocab: set[str]) -> float:
    total = 0
    oov = 0
    for tokens in sentences:
        for token in tokens:
            total += 1
            if token not in vocab:
                oov += 1
    if total == 0:
        return 0.0
    return oov / total


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate n-gram perplexity on validation corpus.")
    parser.add_argument("--model", required=True, help="Path to trained n-gram model.")
    parser.add_argument("--corpus", required=True, help="Validation corpus directory/file.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    model = NGramModel.load(Path(args.model))
    sentences = load_corpus(Path(args.corpus))
    perplexity = model.perplexity(sentences)
    oov = _oov_ratio(sentences, model.vocab)
    log.info(
        json.dumps(
            {
                "event": "text_eval_complete",
                "ts": time.time(),
                "meta": {"perplexity": perplexity, "oov_ratio": oov},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
