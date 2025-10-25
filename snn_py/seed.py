"""Deterministic, named random streams."""

from __future__ import annotations

import random
import hashlib
from typing import Dict


class SeedManager:
    """
    从 base seed + name 派生稳定子种子，返回独立 random.Random 实例。
    """

    def __init__(self, base: int | None):
        self.base = base
        self._cache: Dict[str, random.Random] = {}
        self._seeds: Dict[str, int] = {}

    def _derive(self, name: str) -> int:
        val = self.base if self.base is not None else 0x5DEECE66D
        s = f"{val}:{name}".encode("utf-8")
        h = hashlib.sha256(s).digest()
        # 取前 8 字节作为 64-bit 整数
        return int.from_bytes(h[:8], "big", signed=False)

    def rng(self, name: str) -> random.Random:
        if name not in self._cache:
            seed = self._derive(name)
            self._cache[name] = random.Random(seed)
            self._seeds[name] = seed
        return self._cache[name]

    def seed_for(self, name: str) -> int:
        """Return the numeric seed assigned to `name`."""
        self.rng(name)
        return self._seeds[name]

    def describe(self) -> Dict[str, int]:
        """Expose stream→seed mapping for manifests/logging."""
        return dict(self._seeds)


__all__ = ["SeedManager"]
