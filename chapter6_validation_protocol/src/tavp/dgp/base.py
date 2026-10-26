"""Common interface of the ground-truth data-generating processes (oracles)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np

from tavp.utils import TRADING_DAYS, stable_hash


@dataclass(frozen=True)
class DGP(ABC):
    """A stationary process of daily log-returns with known law.

    Every simulation starts from the stationary distribution of the latent
    state, so that contexts, training histories and scenarios are all draws
    from the same stationary law (Mode S of the protocol).
    """

    dt: float = 1.0 / TRADING_DAYS

    @property
    @abstractmethod
    def name(self) -> str: ...

    def params(self) -> dict:
        return {"name": self.name, **asdict(self)}

    def hash(self) -> str:
        return stable_hash(self.params())

    @abstractmethod
    def simulate_returns(self, n: int, T: int, rng: np.random.Generator) -> np.ndarray:
        """Array (n, T) of log-returns starting from the stationary state."""

    @abstractmethod
    def drift_per_step(self) -> float:
        """Stationary mean of the one-step log-return."""

    def perturbed(self, kind: str, delta: float) -> "DGP":
        """Parametric perturbation of severity delta in [0, 1] (P2, P3, P4)."""
        raise NotImplementedError(f"{kind} is not defined for {self.name}")

    def history(self, T: int, rng: np.random.Generator) -> np.ndarray:
        """A single long path of T log-returns."""
        return self.simulate_returns(1, T, rng)[0]
