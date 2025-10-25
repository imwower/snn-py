"""演示 CLI：合成群体率→意向门→审计→情景记忆。"""

from __future__ import annotations

import argparse
import cProfile
import math
import os
import trace
from pathlib import Path
from random import Random
from typing import List, Mapping, Optional, Sequence, Tuple

from snn_py import logging_config
from snn_py.core import metrics as core_metrics
from snn_py.intent.scoring import NoveltyScorer
from snn_py.manifest import Manifest, save_manifest
from snn_py.memory.episodic import Episode, EpisodicStore
from snn_py.policy.auditor import Auditor
from snn_py.seed import SeedManager


def _parse_args(args: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="snn_py demo run")
    parser.add_argument("--T", type=float, default=8.0, help="模拟时长（秒）")
    parser.add_argument("--policy", required=True, help="策略 JSON 文件")
    parser.add_argument("--jsonl-dir", default=None, help="情景落盘目录")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument("--profile", default=None, help="保存 cProfile 结果到文件")
    parser.add_argument("--trace-coverage", default=None, help="输出 trace 覆盖目录")
    return parser.parse_args(args=args)


def _seed_base(run_id: str, env: Mapping[str, str]) -> int:
    seed_str = env.get("SNN_PY_SEED")
    if seed_str is not None:
        try:
            return int(seed_str)
        except ValueError:
            pass
    try:
        return int(run_id[:8], 16) & 0xFFFFFFFF
    except ValueError:
        return hash(run_id) & 0xFFFFFFFF


def _synth_rate(duration: float, dt: float, rng: Random) -> List[float]:
    steps = max(1, int(duration / dt))
    rates: List[float] = []
    for i in range(steps):
        t = i * dt
        value = 1.0 + 0.6 * math.sin(2.0 * math.pi * t / max(duration, dt))
        value += 0.2 * math.sin(6.0 * math.pi * t / max(duration, dt))
        value += rng.uniform(-0.05, 0.05)
        rates.append(max(0.0, value))
    return rates


def _segments(rates: Sequence[float], thr_low: float, thr_high: float) -> List[Tuple[int, int, bool]]:
    logger = logging_config.get_logger("snn_py.cli.demo")
    segments: List[Tuple[int, int, bool]] = []
    start = 0
    state_up = False
    for idx, value in enumerate(rates):
        if not state_up and value >= thr_high:
            if idx > start:
                segments.append((start, idx, False))
            state_up = True
            start = idx
            logger.info("", extra={"event": "up_start", "meta": {"index": idx, "rate": value}})
        elif state_up and value <= thr_low:
            if idx > start:
                segments.append((start, idx, True))
            state_up = False
            start = idx
            logger.info("", extra={"event": "down_start", "meta": {"index": idx, "rate": value}})
    end_idx = len(rates)
    if end_idx > start:
        segments.append((start, end_idx, state_up))
    return segments


class SimpleIntentGate:
    def __init__(
        self,
        dt: float,
        threshold: float = 0.9,
        decay: float = 0.7,
        refractory: float = 0.15,
        rng: Optional[Random] = None,
    ) -> None:
        self._dt = dt
        self._threshold = threshold
        self._decay = decay
        self._refrac_steps = max(1, int(refractory / dt))
        self._refrac = 0
        self._potential = 0.0
        self._time = 0.0
        self._rng = rng

    def step(self, drive: float) -> Tuple[bool, float]:
        fired = False
        if self._refrac > 0:
            self._refrac -= 1
        else:
            noise = self._rng.uniform(-0.02, 0.02) if self._rng is not None else 0.0
            self._potential = self._potential * self._decay + drive + noise
            if self._potential >= self._threshold:
                fired = True
                self._potential = 0.0
                self._refrac = self._refrac_steps
        self._time += self._dt
        return fired, self._time


def _run(parsed: argparse.Namespace, raw_args: Optional[Sequence[str]] = None) -> int:
    previous_level = os.environ.get("SNN_PY_LOGLEVEL")
    os.environ["SNN_PY_LOGLEVEL"] = parsed.loglevel
    logging_config.setup()
    logger = logging_config.get_logger("snn_py.cli.demo")

    manifest = Manifest.build(Path(parsed.policy), os.environ)
    if raw_args is not None:
        manifest.argv = list(raw_args)
    run_dir = Path("episodes") / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_base = _seed_base(manifest.run_id, os.environ)
    seed_manager = SeedManager(seed_base)
    segments_rng = seed_manager.rng("segments")
    gate_rng = seed_manager.rng("gate")
    logger.info("", extra={"event": "seed_streams_ready", "meta": {"streams": sorted(seed_manager.describe().keys())}})

    jsonl_dir = Path(parsed.jsonl_dir) if parsed.jsonl_dir else run_dir
    store = EpisodicStore(jsonl_dir=jsonl_dir)
    auditor = Auditor(parsed.policy)
    scorer = NoveltyScorer()

    dt = 0.05
    rates = _synth_rate(parsed.T, dt, segments_rng)
    global_mean = sum(rates) / len(rates)
    thr_low = global_mean - 0.1
    thr_high = global_mean + 0.1
    segs = _segments(rates, thr_low, thr_high)

    gate = SimpleIntentGate(dt=dt, threshold=0.85, decay=0.65, refractory=0.1, rng=gate_rng)
    episode_count = 0
    spike_series: List[int] = []
    event_track: List[str] = []

    for idx, (start, end, is_up) in enumerate(segs):
        segment_rates = rates[start:end]
        if not segment_rates:
            continue
        mean_rate = sum(segment_rates) / len(segment_rates)
        q = scorer.score(mean_rate)
        drive = q if is_up else 0.5 * q

        for _rate in segment_rates:
            fired, timestamp = gate.step(drive)
            spike_series.append(1 if fired else 0)
            if not fired:
                continue
            logger.info(
                "",
                extra={
                    "event": "intent_fired",
                    "meta": {"t": round(timestamp, 3), "q": drive, "segment": idx, "up": is_up},
                },
            )
            proposal = {"tool": "ProposeAction", "args": {"note": "spontaneous"}}
            episode = Episode(
                t0=timestamp,
                t1=timestamp,
                kind="proposal",
                meta={"segment": idx, "up": is_up, "q": drive},
                payload=proposal,
            )
            store.append(episode)
            episode_count += 1
            event_track.append("intent_fired")
            result = auditor.check(proposal["tool"], proposal["args"])
            event_track.append("audit_decision")
            if result.status.upper() == "DENIED":
                event_track.append("denied")

    store.close()

    if spike_series:
        metric_win = min(len(spike_series), max(1, int(0.5 / dt)))
    else:
        metric_win = 1
    pop_rate = core_metrics.population_rate([spike_series], dt=dt, win=metric_win)
    count_windows = core_metrics.window_counts(spike_series, metric_win) if spike_series else []
    fano = core_metrics.fano_factor(count_windows)
    stability_stats = core_metrics.stability(rates)
    reliability_stats = core_metrics.reliability(event_track)
    metrics_summary = {
        "window": metric_win,
        "population_rate_len": len(pop_rate),
        "stability": stability_stats,
        "fano": fano,
        "reliability": reliability_stats,
    }
    manifest.artifacts["metrics"] = metrics_summary

    manifest.artifacts["run_dir"] = str(run_dir.resolve())
    manifest.artifacts["episodes_dir"] = str(jsonl_dir.resolve())
    store_run_id = getattr(store, "_run_id", None)
    pattern = f"{store_run_id}.part*.jsonl" if store_run_id else "*.jsonl"
    files = sorted(jsonl_dir.glob(pattern))
    if files:
        serialized = [str(path.resolve()) for path in files]
        manifest.artifacts["episodes_jsonl"] = serialized[0] if len(serialized) == 1 else serialized

    if parsed.profile:
        logger.info("", extra={"event": "profile_saved", "meta": {"path": parsed.profile}})
        manifest.artifacts["profile"] = str(Path(parsed.profile).resolve())
    if parsed.trace_coverage:
        logger.info("", extra={"event": "trace_coverage_done", "meta": {"dir": parsed.trace_coverage}})
        manifest.artifacts["trace_coverage_dir"] = str(Path(parsed.trace_coverage).resolve())
    logger.info(
        "",
        extra={
            "event": "metrics_done",
            "meta": {
                "len": len(pop_rate),
                "cv": stability_stats.get("cv"),
                "fano": fano,
            },
        },
    )
    logger.info("", extra={"event": "run_complete", "meta": {"episodes": episode_count}})

    manifest.seeds = seed_manager.describe()
    manifest_path = save_manifest(manifest, run_dir)
    logger.info("", extra={"event": "manifest_saved", "meta": {"path": str(manifest_path.resolve())}})

    if previous_level is None:
        os.environ.pop("SNN_PY_LOGLEVEL", None)
    else:
        os.environ["SNN_PY_LOGLEVEL"] = previous_level
    logging_config.setup()
    return 0

def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(args)
    raw_args = list(args) if args is not None else None
    if parsed.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        exit_code = _run(parsed, raw_args)
        profiler.disable()
        profiler.dump_stats(parsed.profile)
        return exit_code
    if parsed.trace_coverage:
        tracer = trace.Trace(count=True, trace=False)
        tracer.runfunc(_run, parsed, raw_args)
        results = tracer.results()
        cover_dir = Path(parsed.trace_coverage)
        cover_dir.mkdir(parents=True, exist_ok=True)
        results.write_results(show_missing=True, coverdir=str(cover_dir))
        return 0
    return _run(parsed, raw_args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
