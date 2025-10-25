"""Corpus watcher CLI: rebuild/update n-gram models when files change."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

try:
    from snn_py import logging_config as _logging_config
except Exception:  # pragma: no cover - fallback when logging config missing
    _logging_config = None  # type: ignore[assignment]

from snn_py.text.corpus import load_corpus
from snn_py.text.corpus_watch import changed, dir_fingerprint, load_fp, save_fp
from snn_py.text.ngram import NGramConfig, NGramModel

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


log = _get_logger("snn_py.cli.learn_watch")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train/update n-gram model when corpus changes.")
    parser.add_argument("--corpus", required=True, help="Directory containing *.txt / *.txt.gz files.")
    parser.add_argument("--state", required=True, help="Fingerprint state file for detecting changes.")
    parser.add_argument("--out", required=True, help="Destination model path.")
    parser.add_argument("--order", type=int, default=3, help="N-gram order (default: 3).")
    parser.add_argument("--k", type=float, default=0.5, help="Additive smoothing constant.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    corpus_dir = Path(args.corpus)
    state_path = Path(args.state)
    out_path = Path(args.out)

    fp_old = load_fp(state_path)
    fp_new = dir_fingerprint(corpus_dir)

    if not changed(fp_old, fp_new) and out_path.exists():
        log.info(json.dumps({"event": "learn_watch_skip", "meta": {"reason": "no_change"}}, ensure_ascii=False))
        return 0

    sentences = load_corpus(corpus_dir)
    if out_path.exists():
        model = NGramModel.load(out_path)
        model.update(sentences)
    else:
        config = NGramConfig(order=args.order, k=args.k)
        model = NGramModel(config)
        model.fit(sentences)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    save_fp(state_path, fp_new)
    log.info(
        json.dumps(
            {"event": "learn_watch_done", "meta": {"out": str(out_path), "vocab": len(model.vocab)}},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

