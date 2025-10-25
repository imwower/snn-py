"""Deterministic, named random streams."""

from __future__ import annotations

import random
from typing import Dict, Optional


class SeedManager:
    """Create reproducible Random streams derived from a base seed."""

    def __init__(self, base: Optional[int]) -> None:
        self._base = base
        self._streams: Dict[str, random.Random] = {}
        self._seeds: Dict[str, int] = {}

    def _derive_seed(self, name: str) -> int:
        return hash((self._base, name)) & 0xFFFFFFFF

    def rng(self, name: str) -> random.Random:
        """Return (and cache) a Random instance scoped to `name`."""
        if name not in self._streams:
            seed = self._derive_seed(name)
            self._streams[name] = random.Random(seed)
            self._seeds[name] = seed
        return self._streams[name]

    def seed_for(self, name: str) -> int:
        """Expose the numeric seed for a named stream."""
        self.rng(name)
        return self._seeds[name]

    def describe(self) -> Dict[str, int]:
        """Return a copy of stream→seed mapping for logging/manifests."""
        return dict(self._seeds)


__all__ = ["SeedManager"]
