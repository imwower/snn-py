"""参数扫面 IntentGate 行为。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from snn_py import logging_config
from snn_py.intent.gate import GateConfig, IntentGate
from snn_py.intent.scoring import NoveltyScorer


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IntentGate 参数扫面")
    parser.add_argument("--seeds", required=True, help="逗号分隔的随机种子列表")
    parser.add_argument("--theta", required=True, help="阈值列表或范围，如 0.7:1.0:0.1")
    parser.add_argument("--decay", required=True, help="衰减参数列表，逗号分隔")
    parser.add_argument("--T", type=float, default=5.0, help="仿真时长（秒）")
    parser.add_argument("--dt", type=float, default=0.05, help="时间步长（秒）")
    parser.add_argument("--output", default="results.json", help="输出 JSON 文件路径")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args(argv)


def _parse_seed_list(spec: str) -> List[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def _parse_decay_list(spec: str) -> List[float]:
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def _parse_theta(spec: str) -> List[float]:
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("theta 范围需形如 start:stop:step")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("theta 步长必须为正")
        values: List[float] = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def _drive_signal(num_steps: int) -> Iterable[float]:
    for idx in range(num_steps):
        segment = (idx // 12) % 2
        base = 1.0 if segment == 0 else 0.4
        modulation = 0.2 * math.sin(0.5 * idx)
        yield base + modulation


def _run_case(seed: int, theta: float, decay: float, steps: int, dt: float) -> Dict[str, float]:
    cfg = GateConfig(
        dt=dt,
        lam=decay,
        alpha=1.0,
        sigma=0.0,
        theta=theta,
        refractory=0.0,
        max_rate_hz=10.0,
    )
    gate = IntentGate(cfg, seed=seed)
    scorer = NoveltyScorer(history_len=16)

    fire_times: List[float] = []
    fires = 0
    rate_limit_hits = 0

    original_emit = gate._emit  # type: ignore[attr-defined]

    def _instrument(event: str, q_t: float, since_last: float) -> None:
        nonlocal rate_limit_hits
        if event == "速率限制":
            rate_limit_hits += 1
        original_emit(event, q_t, since_last)

    gate._emit = _instrument  # type: ignore[attr-defined]

    for q in _drive_signal(steps):
        score = scorer.score(q)
        fired, _ = gate.step(score)
        if fired:
            fires += 1
            fire_times.append(gate._time)  # type: ignore[attr-defined]

    gate._emit = original_emit  # type: ignore[attr-defined]

    isi: float
    if len(fire_times) > 1:
        intervals = [fire_times[i] - fire_times[i - 1] for i in range(1, len(fire_times))]
        isi = float(fmean(intervals))
    else:
        isi = 0.0

    return {"fires": float(fires), "mean_isi": isi, "rate_limit_hits": float(rate_limit_hits)}


def _run_sweep(args: argparse.Namespace) -> Dict[str, object]:
    logging_config.setup()
    logger = logging_config.get_logger("snn_py.cli.sweep")

    seeds = _parse_seed_list(args.seeds)
    thetas = _parse_theta(args.theta)
    decays = _parse_decay_list(args.decay)
    dt = args.dt
    total_steps = max(1, int(round(args.T / dt)))

    cases: List[Dict[str, object]] = []

    for seed, theta, decay in itertools.product(seeds, thetas, decays):
        metrics = _run_case(seed, theta, decay, total_steps, dt)
        case = {
            "seed": seed,
            "theta": theta,
            "decay": decay,
            "metrics": metrics,
        }
        cases.append(case)
        logger.info(
            "",
            extra={
                "event": "sweep_case_done",
                "meta": {"seed": seed, "theta": theta, "fires": metrics["fires"]},
            },
        )

    run_context = getattr(logger, "_run", {})
    run_id = run_context.get("run_id")
    output = {"run_id": run_id, "cases": cases}

    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("", extra={"event": "sweep_complete", "meta": {"cases": len(cases)}})
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _run_sweep(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
