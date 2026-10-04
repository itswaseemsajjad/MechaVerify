"""Deterministic, reproducible random sampling.

Every MechaVerify instance is regenerable from a single integer seed. We wrap
``random.Random`` (sufficient for discrete sampling) rather than NumPy's global
state so that no global RNG is ever touched and parallel generation stays
hermetic. Derive child streams with ``SeededRNG.derive`` to keep per-instance
draws independent of generation order.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def _mix(seed: int, label: str) -> int:
    """Stable 64-bit mix of (seed, label) so derived streams are order-independent."""
    h = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


class SeededRNG:
    """A thin, hermetic wrapper over ``random.Random``."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._r = random.Random(self.seed)

    def derive(self, label: str) -> "SeededRNG":
        """Return an independent stream keyed by ``(seed, label)``."""
        return SeededRNG(_mix(self.seed, label))

    # --- primitive draws -------------------------------------------------
    def randint(self, lo: int, hi: int) -> int:
        """Inclusive integer in [lo, hi]."""
        return self._r.randint(lo, hi)

    def uniform(self, lo: float, hi: float) -> float:
        return self._r.uniform(lo, hi)

    def random(self) -> float:
        return self._r.random()

    def choice(self, seq: Sequence[T]) -> T:
        return self._r.choice(seq)

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        return self._r.sample(list(seq), k)

    def shuffle(self, seq: list) -> None:
        self._r.shuffle(seq)

    def chance(self, p: float) -> bool:
        """True with probability ``p``."""
        return self._r.random() < p
