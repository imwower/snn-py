"""Auto-train n-gram model on corpus changes and emit narrations continuously."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Optional

try:
    from snn_py import logging_config as _logging_config
except Exception:  # pragma: no cover - fallback when logging config missing
    _logging_config = None  # type: ignore[assignment]

from snn_py.text.corpus import load_corpus
from snn_py.text.corpus_watch import Fingerprint, changed, dir_fingerprint, load_fp, save_fp
from snn_py.text.ngram import NGramConfig, NGramModel
from snn_py.text.narrator import Narrator

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


log = _get_logger("snn_py.cli.auto_narrate")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Continuously train on corpus updates and emit narrations.")
    parser.add_argument("--corpus", required=True, help="Directory/file containing *.txt / *.txt.gz corpora.")
    parser.add_argument("--state", required=True, help="Fingerprint state file for detecting changes.")
    parser.add_argument("--model", required=True, help="Path to persist the n-gram model.")
    parser.add_argument("--out", required=True, help="Output narration JSONL file.")
    parser.add_argument("--order", type=int, default=3, help="N-gram order (default: 3).")
    parser.add_argument("--k", type=float, default=0.5, help="Additive smoothing constant.")
    parser.add_argument("--poll-ms", type=int, default=500, help="Sleep duration between checks (ms).")
    parser.add_argument("--emit-ms", type=int, default=2000, help="Interval between auto narrations (ms).")
    parser.add_argument("--context", nargs="+", default=["叙述者"], help="Context tokens prepended to each narration.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for randomness.")
    parser.add_argument("--q-min", type=float, default=0.2, help="Minimum q value when sampling narrations.")
    parser.add_argument("--q-max", type=float, default=0.8, help="Maximum q value when sampling narrations.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after emitting N narrations (0 = run indefinitely).",
    )
    return parser.parse_args(argv)


def _fingerprint_target(target: Path) -> Fingerprint:
    target = Path(target)
    if target.is_dir():
        return dir_fingerprint(target)
    if target.is_file():
        try:
            stat = target.stat()
        except OSError:
            return {}
        return {str(target.resolve()): (stat.st_size, stat.st_mtime)}
    return {}


def _train_if_needed(
    *,
    corpus_path: Path,
    state_path: Path,
    model_path: Path,
    fp_previous: Fingerprint,
    order: int,
    k: float,
) -> tuple[bool, Fingerprint]:
    fp_current = _fingerprint_target(corpus_path)
    needs_training = changed(fp_previous, fp_current) or not model_path.exists()
    if not needs_training:
        return False, fp_previous

    sentences = load_corpus(corpus_path)
    if not sentences:
        log.info(json.dumps({"event": "auto_train_skip", "meta": {"reason": "empty_corpus"}}, ensure_ascii=False))
        return False, fp_previous

    if model_path.exists():
        model = NGramModel.load(model_path)
        model.update(sentences)
    else:
        config = NGramConfig(order=order, k=k)
        model = NGramModel(config)
        model.fit(sentences)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_fp(state_path, fp_current)
    log.info(
        json.dumps(
            {"event": "auto_train_complete", "meta": {"model": str(model_path), "vocab": len(model.vocab)}},
            ensure_ascii=False,
        )
    )
    return True, fp_current


def _write_narration(out_path: Path, event: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    corpus_path = Path(args.corpus)
    state_path = Path(args.state)
    model_path = Path(args.model)
    out_path = Path(args.out)

    poll_interval = max(0, args.poll_ms) / 1000.0
    emit_interval = max(0.1, args.emit_ms / 1000.0)
    rng = random.Random(args.seed)
    context_tokens = args.context or []

    fp_cached = load_fp(state_path)
    narrator: Optional[Narrator] = None
    emitted = 0
    next_emit_ts = time.time()

    log.info(
        json.dumps(
            {
                "event": "auto_loop_start",
                "meta": {"corpus": str(corpus_path), "model": str(model_path), "out": str(out_path)},
            },
            ensure_ascii=False,
        )
    )

    while True:
        trained, fp_cached = _train_if_needed(
            corpus_path=corpus_path,
            state_path=state_path,
            model_path=model_path,
            fp_previous=fp_cached,
            order=args.order,
            k=args.k,
        )
        model_ready = model_path.exists()
        if trained or (narrator is None and model_ready):
            narrator = Narrator(model_path, seed=args.seed)

        now = time.time()
        should_emit = now >= next_emit_ts and narrator is not None
        if should_emit:
            q = rng.uniform(min(args.q_min, args.q_max), max(args.q_min, args.q_max))
            narration = narrator.narrate(q=q, context=context_tokens)
            record = {
                "event": "auto_narration",
                "ts": now,
                "meta": {"q": round(q, 3), "text": narration.text, "explain": narration.explain},
            }
            _write_narration(out_path, record)
            log.info(json.dumps(record, ensure_ascii=False))
            emitted += 1
            next_emit_ts = now + emit_interval
            if args.max_events and emitted >= args.max_events:
                log.info(
                    json.dumps(
                        {"event": "auto_loop_complete", "meta": {"emitted": emitted}},
                        ensure_ascii=False,
                    )
                )
                return 0

        time.sleep(poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
