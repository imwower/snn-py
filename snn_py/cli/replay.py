"""Replay CLI for episodic JSONL archives."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple

from snn_py import logging_config
from snn_py.intent.scoring import NoveltyScorer
from snn_py.memory.episodic import Episode

logger = logging_config.get_logger("snn_py.cli.replay")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay episodic JSONL runs.")
    parser.add_argument(
        "--jsonl-dir",
        required=True,
        help="Directory containing *.jsonl or *.jsonl.gz archives.",
    )
    parser.add_argument(
        "--since",
        type=float,
        default=None,
        help="Minimum timestamp (seconds) to include based on episode t1.",
    )
    parser.add_argument(
        "--time-travel",
        action="store_true",
        help="Recompute expected firing frames based on manifest + seeds.",
    )
    parser.add_argument(
        "--expect",
        default=None,
        help="Path to a JSON file containing expected replay summary stats.",
    )
    return parser.parse_args(argv)


def _iter_files(jsonl_dir: Path) -> Iterator[Path]:
    patterns = ("*.jsonl", "*.jsonl.gz")
    candidates = []
    for pattern in patterns:
        candidates.extend(jsonl_dir.glob(pattern))
    for path in sorted(candidates):
        if path.is_file():
            yield path


def _extract_run_id(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    if ".part" in name:
        return name.split(".part", 1)[0]
    return name


def _open_jsonl(path: Path) -> TextIO:
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_episodes(files: Iterable[Path], since: Optional[float]) -> Tuple[Dict[str, List[Episode]], Dict[str, Path]]:
    grouped: Dict[str, List[Episode]] = defaultdict(list)
    sources: Dict[str, Path] = {}
    for path in files:
        run_id = _extract_run_id(path)
        sources.setdefault(run_id, path)
        try:
            with _open_jsonl(path) as handle:
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
    return grouped, sources


def _episode_to_span(episode: Episode) -> Dict[str, object]:
    span: Dict[str, object] = {"t0": episode.t0, "t1": episode.t1, "kind": episode.kind}
    if episode.meta:
        span["meta"] = episode.meta
    if episode.payload:
        span["payload"] = episode.payload
    return span


def _extract_duration(argv: Sequence[str], default: float = 8.0) -> float:
    for idx, token in enumerate(argv):
        if token == "--T" and idx + 1 < len(argv):
            try:
                return float(argv[idx + 1])
            except ValueError:
                return default
    return default


def _extract_dt(manifest: Dict[str, object], default: float = 0.05) -> float:
    env = manifest.get("env_whitelist")
    if isinstance(env, dict):
        dt_str = env.get("SNN_PY_DT")
        if dt_str is not None:
            try:
                return float(dt_str)
            except ValueError:
                return default
    return default


def _synth_rate(duration: float, dt: float, rng: Random) -> List[float]:
    steps = max(1, int(duration / max(dt, 1e-9)))
    rates: List[float] = []
    for i in range(steps):
        t = i * dt
        base = 1.0 + 0.6 * math.sin(2.0 * math.pi * t / max(duration, dt))
        base += 0.2 * math.sin(6.0 * math.pi * t / max(duration, dt))
        base += rng.uniform(-0.05, 0.05)
        rates.append(max(0.0, base))
    return rates


def _segments(rates: Sequence[float], thr_low: float, thr_high: float) -> List[Tuple[int, int, bool]]:
    segments: List[Tuple[int, int, bool]] = []
    start = 0
    state_up = False
    for idx, value in enumerate(rates):
        if not state_up and value >= thr_high:
            if idx > start:
                segments.append((start, idx, False))
            state_up = True
            start = idx
        elif state_up and value <= thr_low:
            if idx > start:
                segments.append((start, idx, True))
            state_up = False
            start = idx
    end_idx = len(rates)
    if end_idx > start:
        segments.append((start, end_idx, state_up))
    return segments


class _TravelGate:
    def __init__(self, dt: float, rng: Random, threshold: float = 0.85, decay: float = 0.65, refractory: float = 0.1) -> None:
        self._dt = dt
        self._rng = rng
        self._threshold = threshold
        self._decay = decay
        self._refrac_steps = max(1, int(refractory / max(dt, 1e-9)))
        self._refrac = 0
        self._potential = 0.0
        self._time = 0.0

    def step(self, drive: float) -> Tuple[bool, float]:
        fired = False
        if self._refrac > 0:
            self._refrac -= 1
        else:
            noise = self._rng.uniform(-0.02, 0.02)
            self._potential = self._potential * self._decay + drive + noise
            if self._potential >= self._threshold:
                fired = True
                self._potential = 0.0
                self._refrac = self._refrac_steps
        self._time += self._dt
        return fired, self._time


def _recompute_frames(manifest: Dict[str, object]) -> Optional[List[int]]:
    seeds = manifest.get("seeds")
    if not isinstance(seeds, dict):
        return None
    try:
        segments_seed = int(seeds["segments"])
        gate_seed = int(seeds["gate"])
    except (KeyError, TypeError, ValueError):
        return None

    argv = manifest.get("argv") or []
    duration = _extract_duration(argv)
    dt = _extract_dt(manifest)

    rates = _synth_rate(duration, dt, Random(segments_seed))
    if not rates:
        return []
    global_mean = sum(rates) / len(rates)
    thr_low = global_mean - 0.1
    thr_high = global_mean + 0.1
    segs = _segments(rates, thr_low, thr_high)
    scorer = NoveltyScorer()
    gate = _TravelGate(dt=dt, rng=Random(gate_seed))
    frames: List[int] = []
    step_idx = 0
    for start, end, is_up in segs:
        segment_rates = rates[start:end]
        if not segment_rates:
            continue
        mean_rate = sum(segment_rates) / len(segment_rates)
        q = scorer.score(mean_rate)
        drive = q if is_up else 0.5 * q
        for _ in segment_rates:
            fired, _ = gate.step(drive)
            if fired:
                frames.append(step_idx)
            step_idx += 1
    return frames


def _episodes_to_frames(episodes: List[Episode], dt: float) -> List[int]:
    if not episodes:
        return []
    frames = []
    for episode in episodes:
        if episode.kind != "proposal":
            continue
        try:
            frame = int(round(episode.t0 / max(dt, 1e-9)))
        except (TypeError, ValueError):
            continue
        frames.append(frame)
    return frames


def _frame_match_rate(actual: List[int], expected: List[int]) -> float:
    if not actual and not expected:
        return 1.0
    actual_set = set(actual)
    matches = sum(1 for frame in expected if frame in actual_set)
    denom = max(len(actual), len(expected), 1)
    return matches / denom


def _resolve_manifest_path(jsonl_dir: Path, run_id: str, source: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if source is not None:
        candidates.append(source.parent / "manifest.json")
    candidates.append(jsonl_dir / run_id / "manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _perform_time_travel(
    jsonl_dir: Path,
    grouped: Dict[str, List[Episode]],
    sources: Dict[str, Path],
) -> None:
    for run_id, episodes in grouped.items():
        manifest_path = _resolve_manifest_path(jsonl_dir, run_id, sources.get(run_id))
        if manifest_path is None:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected_frames = _recompute_frames(manifest)
        if expected_frames is None:
            continue
        dt = _extract_dt(manifest)
        actual_frames = _episodes_to_frames(episodes, dt)
        match_rate = _frame_match_rate(actual_frames, expected_frames)
        logger.info(
            "",
            extra={"event": "replay_recomputed", "meta": {"run_id": run_id, "match_rate": match_rate}},
        )


def _load_expectation(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def _diff_summary(expected: Dict[str, object], actual: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    mismatch: Dict[str, Dict[str, object]] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatch[key] = {"expected": expected_value, "actual": actual_value}
    return mismatch


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(argv)
    logging_config.setup()

    jsonl_dir = Path(parsed.jsonl_dir)
    files = list(_iter_files(jsonl_dir))
    grouped, sources = _load_episodes(files, parsed.since)

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

    if parsed.time_travel:
        _perform_time_travel(jsonl_dir, grouped, sources)

    if parsed.expect:
        expect_path = Path(parsed.expect)
        expected = _load_expectation(expect_path)
        if expected is not None:
            mismatch = _diff_summary(expected, summary)
            logger.info("", extra={"event": "replay_diff", "meta": {"mismatch": mismatch}})
        else:
            logger.warning(
                "unable to load expectation file",
                extra={"event": "replay_diff", "meta": {"mismatch": {"error": "unreadable"}}},
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
