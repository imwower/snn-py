"""Simple additive-smoothed n-gram language model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math


@dataclass(slots=True)
class NGramConfig:
    order: int = 3
    k: float = 0.5
    lowercase: bool = True
    start_token: str = "<s>"
    end_token: str = "</s>"

    def __post_init__(self) -> None:
        if self.order < 2:
            raise ValueError("order must be >= 2 for context to exist")
        if self.k < 0:
            raise ValueError("k must be non-negative")


class NGramModel:
    def __init__(self, config: NGramConfig | None = None):
        self.config = config or NGramConfig()
        self._context_size = self.config.order - 1
        self._counts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._vocab: set[str] = set()

    @property
    def context_size(self) -> int:
        return self._context_size

    @property
    def vocab(self) -> set[str]:
        return set(self._vocab)

    def _normalize(self, token: str) -> str:
        return token.lower() if self.config.lowercase else token

    def _reset(self) -> None:
        self._counts = defaultdict(Counter)
        self._vocab = set()

    def _ingest(self, corpus: Sequence[Sequence[str]]) -> None:
        pad = [self.config.start_token] * self._context_size
        for sentence in corpus:
            tokens = [self._normalize(tok) for tok in sentence if tok]
            if not tokens:
                continue
            seq = pad + tokens + [self.config.end_token]
            for idx in range(self._context_size, len(seq)):
                context = tuple(seq[idx - self._context_size : idx])
                target = seq[idx]
                self._counts[context][target] += 1
                if target != self.config.end_token:
                    self._vocab.add(target)

    def fit(self, corpus: Sequence[Sequence[str]]) -> None:
        """Build n-gram statistics from scratch using `corpus`."""
        self._reset()
        self._ingest(corpus)

    def update(self, corpus: Sequence[Sequence[str]]) -> None:
        """Incrementally update statistics with additional sentences."""
        if not isinstance(self._counts, defaultdict):
            self._counts = defaultdict(Counter, self._counts)
        self._ingest(corpus)

    def _sample(self, context: Tuple[str, ...], rng: Random) -> str:
        distribution = self._counts.get(context)
        tokens: List[str]
        weights: List[float]
        if distribution:
            tokens = list(distribution.keys())
            weights = [count + self.config.k for count in distribution.values()]
            if self.config.end_token not in distribution:
                tokens.append(self.config.end_token)
                weights.append(max(self.config.k, 1e-3))
        else:
            tokens = list(self._vocab) if self._vocab else []
            weights = [1.0] * len(tokens)
            tokens.append(self.config.end_token)
            weights.append(1.0)
        total = sum(weights)
        if total <= 0:
            return self.config.end_token
        pick = rng.random() * total
        upto = 0.0
        for token, weight in zip(tokens, weights):
            upto += weight
            if pick <= upto:
                return token
        return tokens[-1]

    def _prob(self, context: Tuple[str, ...], token: str) -> float:
        distribution = self._counts.get(context)
        if not distribution:
            vocab_size = len(self._vocab) or 1
            return 1.0 / (vocab_size + 1)
        total = sum(distribution.values()) + self.config.k * (len(distribution) + 1)
        count = distribution.get(token, 0)
        return max(1e-12, (count + self.config.k) / total)

    def evaluate(self, corpus: Sequence[Sequence[str]]) -> Tuple[float, int, int]:
        """Return (neg_log_sum, token_count, sentence_count) for a corpus."""
        pad = [self.config.start_token] * self._context_size
        neg_log_sum = 0.0
        token_count = 0
        sent_count = 0
        for sentence in corpus:
            tokens = [self._normalize(tok) for tok in sentence if tok]
            if not tokens:
                continue
            seq = pad + tokens + [self.config.end_token]
            for idx in range(self._context_size, len(seq)):
                context = tuple(seq[idx - self._context_size : idx])
                target = seq[idx]
                prob = self._prob(context, target)
                neg_log_sum += -math.log(prob)
                token_count += 1
            sent_count += 1
        return neg_log_sum, token_count, sent_count

    def perplexity(self, corpus: Sequence[Sequence[str]]) -> float:
        """Compute empirical perplexity on `corpus`."""
        neg_log_sum, token_count, _ = self.evaluate(corpus)
        if token_count == 0:
            return float("inf")
        return math.exp(neg_log_sum / token_count)

    def generate(
        self,
        max_tokens: int = 30,
        *,
        rng: Optional[Random] = None,
        prefix: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate up to `max_tokens` tokens conditioned on an optional prefix."""
        if max_tokens <= 0:
            return []
        rng = rng or Random()
        prefix_tokens = [self._normalize(tok) for tok in (prefix or []) if tok]
        context = [self.config.start_token] * self._context_size
        if prefix_tokens:
            merged = context + prefix_tokens
            context = merged[-self._context_size :]
        result: List[str] = []
        while len(result) < max_tokens:
            ctx_tuple = tuple(context[-self._context_size :]) if self._context_size else tuple()
            token = self._sample(ctx_tuple, rng)
            if token == self.config.end_token:
                break
            result.append(token)
            context.append(token)
        return result

    def save(self, path: Path | str) -> None:
        """Serialize the model to disk."""
        payload = {
            "config": asdict(self.config),
            "counts": [
                {"context": list(context), "next": dict(counter)}
                for context, counter in self._counts.items()
            ],
            "vocab": sorted(self._vocab),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "NGramModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = NGramConfig(**data["config"])
        model = cls(config)
        counts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for bucket in data.get("counts", []):
            context = tuple(bucket.get("context", []))
            next_counts = Counter({tok: int(cnt) for tok, cnt in bucket.get("next", {}).items()})
            counts[context] = next_counts
        model._counts = counts
        model._vocab = set(data.get("vocab", []))
        return model


__all__ = ["NGramModel", "NGramConfig"]
