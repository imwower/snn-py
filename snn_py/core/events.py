"""群体活动的事件检测工具。"""

from __future__ import annotations

import json
import math
import statistics
import time
from typing import Dict, Iterable, List, Optional, Tuple

from snn_py import logging_config
from snn_py.event_schema import EVENT_CODES


_LOGGER = logging_config.get_logger("snn_py.core.events")


def _emit(event: str, **meta: float) -> None:
    payload = {"event": event, "ts": time.time(), "meta": meta}
    code = EVENT_CODES.get(event)
    if code is not None:
        payload["code"] = code
    _LOGGER.info(json.dumps(payload, separators=(",", ":")))


def detect_up_down(
    pop_rate: List[float], thr_low: float, thr_high: float
) -> List[Tuple[int, int, bool]]:
    """使用滞后阈值将群体放电率切分为上/下状态段。"""
    if thr_low > thr_high:
        raise ValueError("低阈值必须小于等于高阈值")
    segments: List[Tuple[int, int, bool]] = []
    if not pop_rate:
        return segments

    state_up = False
    current_start = 0

    for idx, value in enumerate(pop_rate):
        if state_up:
            if value <= thr_low:
                segments.append((current_start, idx, True))
                state_up = False
                current_start = idx
                _emit("下状态开始", start=idx)
        else:
            if value >= thr_high:
                segments.append((current_start, idx, False))
                state_up = True
                current_start = idx
                _emit("上状态开始", start=idx)

    segments.append((current_start, len(pop_rate), state_up))
    return segments


def _compute_bin_width(spike_times: List[int]) -> Optional[int]:
    if len(spike_times) <= 1:
        return 1 if spike_times else None
    diffs = [b - a for a, b in zip(spike_times, spike_times[1:]) if b > a]
    if not diffs:
        return 1
    avg = statistics.fmean(diffs)
    return max(1, int(round(avg)))


def detect_avalanches(spikes: List[List[int]], bin_width: Optional[int] = None) -> List[Dict[str, int]]:
    """通过时间分箱检测神经雪崩。"""
    if not spikes:
        return []
    time_bins = len(spikes)
    width = bin_width
    if width is None:
        spike_times = [t for t, row in enumerate(spikes) if any(row)]
        computed = _compute_bin_width(spike_times)
        if computed is None:
            return []
        width = computed

    if width <= 0:
        raise ValueError("分箱宽度必须为正值")

    num_bins = math.ceil(time_bins / width)
    binned_counts: List[int] = [0] * num_bins
    for t, row in enumerate(spikes):
        bin_idx = min(t // width, num_bins - 1)
        binned_counts[bin_idx] += sum(row)

    avalanches: List[Dict[str, int]] = []
    idx = 0
    while idx < num_bins:
        if binned_counts[idx] == 0:
            idx += 1
            continue
        start = idx
        total = 0
        while idx < num_bins and binned_counts[idx] > 0:
            total += binned_counts[idx]
            idx += 1
        duration = idx - start
        avalanche = {"size": total, "duration_bins": duration}
        avalanches.append(avalanche)
        _emit("神经雪崩", size=total, duration=duration)
    return avalanches
