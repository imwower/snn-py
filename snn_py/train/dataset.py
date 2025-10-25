"""Dataset helpers for simple gate threshold training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

from snn_py.util.jsonl import iter_jsonl, scan_jsonl_dir


@dataclass
class Sample:
    q: float
    y: int  # 1 if proposal emitted, else 0


def _coerce_float(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def load_dataset(jsonl_dir: Path) -> List[Sample]:
    """Load segments/proposals from JSONL directory."""
    seg_q: List[float] = []
    props_idx: Set[int] = set()
    proposal_total = 0
    idx = 0  # 1-based segment index

    for path in scan_jsonl_dir(jsonl_dir):
        for record in iter_jsonl(path):
            event = record.get("event")
            if event == "segment":
                meta = record.get("meta") or {}
                q = _coerce_float(meta.get("q"))
                if q is not None:
                    seg_q.append(q)
                    idx += 1
            elif event == "proposal":
                proposal_total += 1
                if idx > 0:
                    props_idx.add(idx)

    if not seg_q:
        return []

    pos_target = min(proposal_total, len(seg_q))
    if pos_target and len(props_idx) < pos_target:
        # Fallback: proposals were logged out-of-order, approximate by highest-q segments.
        sorted_idx = sorted(range(len(seg_q)), key=lambda i: seg_q[i], reverse=True)[:pos_target]
        props_idx = {i + 1 for i in sorted_idx}

    samples: List[Sample] = []
    for i, q in enumerate(seg_q, start=1):
        y = 1 if i in props_idx else 0
        samples.append(Sample(q=q, y=y))
    return samples


def train_val_split(xs: List[Sample], val_ratio: float = 0.2) -> Tuple[List[Sample], List[Sample]]:
    if not xs:
        return [], []
    if val_ratio < 0.0:
        val_ratio = 0.0
    if val_ratio > 1.0:
        val_ratio = 1.0
    n = len(xs)
    k = max(1, int(n * (1 - val_ratio)))
    if k > n:
        k = n
    return xs[:k], xs[k:]


__all__ = ["Sample", "load_dataset", "train_val_split"]
