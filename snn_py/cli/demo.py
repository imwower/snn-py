"""演示从 LIF 网络到策略审计的端到端流程。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from typing import Iterable, List

from snn_py import logging_config
from snn_py.core import LIF, LIFConfig, detect_up_down
from snn_py.intent import GateConfig, IntentGate, NoveltyScorer
from snn_py.memory import Episode, EpisodicStore
from snn_py.policy import Auditor


def _moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, window)
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def main(args: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="演示主循环：LIF → 审计器")
    parser.add_argument("--T", type=float, default=8.0, help="模拟时长（秒）")
    parser.add_argument("--policy", required=True, help="策略 JSON 路径")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    parsed = parser.parse_args(args)

    os.environ["SNN_PY_LOGLEVEL"] = parsed.loglevel
    logging_config.setup()
    logger = logging_config.get_logger("snn_py.cli.demo")
    logger.setLevel(logging.INFO)

    lif_cfg = LIFConfig(
        n=40,
        frac_inh=0.25,
        p_conn=0.2,
        dt=0.001,
        tau_m=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        w_e=1.5,
        w_i=-1.2,
        refrac_steps=3,
        ext_noise=0.6,
    )
    lif = LIF(lif_cfg, seed=17)
    spikes = lif.run(parsed.T)

    counts = [sum(step) for step in spikes]
    rates = _moving_average(counts, window=50)
    if rates:
        global_mean = statistics.fmean(rates)
    else:
        global_mean = 0.0

    base = global_mean if global_mean > 0 else 1.0
    thr_low = max(0.0, 0.8 * base)
    thr_high = max(thr_low + max(0.1 * base, 0.1), thr_low + 0.1)

    segments = detect_up_down(rates, thr_low=thr_low, thr_high=thr_high)

    scorer = NoveltyScorer()
    gate_cfg = GateConfig(
        dt=0.05,
        lam=0.15,
        alpha=0.4,
        sigma=1.0,
        theta=0.75,
        refractory=0.2,
        max_rate_hz=4.0,
    )
    gate = IntentGate(gate_cfg, seed=11)
    store = EpisodicStore()
    auditor = Auditor(parsed.policy)

    episode_count = 0
    gate_time = 0.0

    lif_dt = lif_cfg.dt
    for seg_index, (start, end, is_up) in enumerate(segments):
        if end <= start:
            continue
        segment_rates = rates[start:end]
        if segment_rates:
            mean_rate = statistics.fmean(segment_rates)
        else:
            mean_rate = 0.0
        x = mean_rate - global_mean
        q_t = scorer.score(x)

        duration = (end - start) * lif_dt
        steps_needed = max(1, int(duration / gate_cfg.dt))

        for _ in range(steps_needed):
            fired, _ = gate.step(q_t)
            gate_time += gate_cfg.dt
            if not fired:
                continue
            logger.info(
                json.dumps(
                    {"event": "proposal", "meta": {"t": gate_time, "q": q_t}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            proposal = {"tool": "ProposeAction", "args": {"note": "spontaneous"}}
            episode = Episode(
                t0=gate_time,
                t1=gate_time,
                kind="proposal",
                meta={"segment": seg_index, "is_up": is_up, "q": q_t},
                payload=proposal,
            )
            store.append(episode)
            episode_count += 1

            result = auditor.check(proposal["tool"], proposal["args"])
            logger.info(
                json.dumps(
                    {"event": "audit_result", "meta": {"status": result.status}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

    payload = {"event": "run_complete", "meta": {"episodes": episode_count}}
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
