"""Replay CLI for episodic JSONL archives."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from snn_py import logging_config
from snn_py.memory.episodic import Episode

logger = logging_config.get_logger("snn_py.cli.replay")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay episodic JSONL runs.")
    parser.add_argument("--jsonl-dir", required=True, help="Directory containing *.jsonl archives.")
    parser.add_argument(
        "--since",
        type=float,
        default=None,
        help="Minimum timestamp (seconds) to include based on episode t1.",
    )
    return parser.parse_args(argv)


def _iter_files(jsonl_dir: Path) -> Iterator[Path]:
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        if path.is_file():
            yield path


def _extract_run_id(path: Path) -> str:
    name = path.stem  # drops .jsonl
    if ".part" in name:
        return name.split(".part", 1)[0]
    return name


def _load_episodes(files: Iterable[Path], since: Optional[float]) -> Dict[str, List[Episode]]:
    grouped: Dict[str, List[Episode]] = defaultdict(list)
    for path in files:
        run_id = _extract_run_id(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("skip malformed line", extra={"event": "replay_skip", "meta": {"file": str(path)}})
                        continue
                    try:
                        episode = Episode(**payload)
                    except TypeError:
                        logger.warning(
                            "skip incompatible episode",
                            extra={"event": "replay_skip", "meta": {"file": str(path), "payload": payload}},
                        )
                        continue
                    if since is not None and episode.t1 < since:
                        continue
                    grouped[run_id].append(episode)
        except OSError:
            logger.warning("unable to read jsonl file", extra={"event": "replay_skip", "meta": {"file": str(path)}})
    return grouped


def _episode_to_span(episode: Episode) -> Dict[str, object]:
    span: Dict[str, object] = {"t0": episode.t0, "t1": episode.t1, "kind": episode.kind}
    if episode.meta:
        span["meta"] = episode.meta
    if episode.payload:
        span["payload"] = episode.payload
    return span


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(argv)
    logging_config.setup()

    jsonl_dir = Path(parsed.jsonl_dir)
    files = list(_iter_files(jsonl_dir))
    grouped = _load_episodes(files, parsed.since)

    total_episodes = sum(len(items) for items in grouped.values())
    logger.info(
        "",
        extra={"event": "replay_loaded", "meta": {"files": len(files), "episodes": total_episodes}},
    )

    for run_id, episodes in grouped.items():
        spans = [_episode_to_span(ep) for ep in sorted(episodes, key=lambda item: (item.t0, item.t1))]
        logger.info(
            "",
            extra={"event": "replay_timeline", "meta": {"run_id": run_id, "spans": spans}},
        )

    proposals = sum(1 for episodes in grouped.values() for ep in episodes if ep.kind == "proposal")
    summary = {"runs": len(grouped), "proposals": proposals}
    logger.info("", extra={"event": "replay_summary", "meta": summary})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
