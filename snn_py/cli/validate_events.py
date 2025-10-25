from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from snn_py.logging_config import get_logger as _get_logger
except Exception:

    def _get_logger(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger(name)


from snn_py.schema.events_schema import validate_stream
from snn_py.util.jsonl import iter_jsonl, scan_jsonl_dir

log = _get_logger("snn_py.cli.validate_events")


def _j(event: str, **meta: Any) -> None:
    rec = {"event": event, "ts": time.time(), "meta": meta}
    log.info(json.dumps(rec, ensure_ascii=False))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser("validate events jsonl")
    parser.add_argument("--jsonl-dir", required=True)
    args = parser.parse_args(argv)

    events: List[Dict[str, Any]] = []
    for path in scan_jsonl_dir(Path(args.jsonl_dir)):
        for payload in iter_jsonl(path):
            events.append(payload)

    stats = validate_stream(events)
    _j("validate_complete", **stats.as_dict())
    return 0 if stats.bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
