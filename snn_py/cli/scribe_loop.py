"""Continuous scribe loop: tail proposal events and emit narrations."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

try:
    from snn_py import logging_config as _logging_config
except Exception:  # pragma: no cover - fallback path
    _logging_config = None  # type: ignore[assignment]

from snn_py.stream.tail_jsonl import JSONLTailer
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


log = _get_logger("snn_py.cli.scribe_loop")


def _j(event: str, **meta: object) -> None:
    rec = {"event": event, "ts": time.time(), "meta": meta}
    log.info(json.dumps(rec, ensure_ascii=False))


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Tail proposals and emit narrated text.")
    parser.add_argument("--in", dest="inp", required=True, help="Input proposals JSONL (append-only).")
    parser.add_argument("--out", required=True, help="Output narration JSONL.")
    parser.add_argument("--state", required=True, help="State file to persist the tail offset.")
    parser.add_argument("--model", required=True, help="Path to a serialized n-gram model.")
    parser.add_argument("--poll-ms", type=int, default=200, help="Sleep duration between polls (ms).")
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after processing N proposal events (0 = run indefinitely).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    tail = JSONLTailer(Path(args.inp), start_at_end=False)
    state_path = Path(args.state)
    tail.load_state(state_path)

    narrator = Narrator(Path(args.model), seed=7)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    poll_interval = max(0, args.poll_ms) / 1000.0
    _j("scribe_loop_start", inp=args.inp, out=args.out, model=args.model)

    try:
        while True:
            batch = tail.read_new(max_lines=1000)
            for event in batch:
                if not isinstance(event, dict):
                    continue
                if event.get("event") != "proposal":
                    continue
                meta = event.get("meta") or {}
                q = meta.get("q", 0.5)

                narration = narrator.narrate(q=q, context=["the", "agent"])
                record = {
                    "event": "narration",
                    "ts": time.time(),
                    "meta": {"q": q, "text": narration.text, "explain": narration.explain},
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                processed += 1
                _j("narration", q=q, text=narration.text)
                if args.max_events and processed >= args.max_events:
                    tail.save_state(state_path)
                    _j("scribe_loop_complete", processed=processed)
                    return 0

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        _j("scribe_loop_interrupt", processed=processed)
    finally:
        tail.save_state(state_path)
        tail.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
