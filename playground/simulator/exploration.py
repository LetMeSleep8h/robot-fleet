"""
Exploration for fleet data collection.

A deterministic subset of robots (by robot_id hash) adds Gaussian noise to the
policy's actions so the recorded fleet data covers more than the deployed
policy's own behaviour. The noise is held constant over one transition
interval, so the recorded action is the one actually applied.

Environment variables:
    EXPLORATION_FRACTION  share of robots that explore (default 0.2)
    EXPLORATION_STD       std of the per-dimension Gaussian noise (default 0.1)
"""

import os
import zlib

import numpy as np

DEFAULT_FRACTION = 0.2
DEFAULT_STD = 0.1
_HASH_BUCKETS = 1000


def is_exploration_robot(robot_id: str, fraction: float) -> bool:
    """Deterministically decide whether a robot explores (stable across restarts)."""
    return zlib.crc32(robot_id.encode()) % _HASH_BUCKETS < fraction * _HASH_BUCKETS


class Explorer:
    """Adds held Gaussian noise to the actions of exploration robots."""

    def __init__(self, fraction: float, std: float, rng: np.random.Generator | None = None):
        self.fraction = fraction
        self.std = std
        self._rng = rng or np.random.default_rng()
        self._noise: dict[str, np.ndarray] = {}

    @classmethod
    def from_env(cls) -> "Explorer":
        return cls(
            fraction=float(os.environ.get("EXPLORATION_FRACTION", DEFAULT_FRACTION)),
            std=float(os.environ.get("EXPLORATION_STD", DEFAULT_STD)),
        )

    def apply(self, robot_id: str, action: np.ndarray, refresh: bool,
              bounds: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, bool]:
        """Return (applied action, explored). refresh=True draws new noise.

        bounds: the robot's per-actuator (low, high) control range; noisy
        actions are clipped to it (the range the policy was trained in).
        """
        if self.std <= 0 or not is_exploration_robot(robot_id, self.fraction):
            return action, False
        if refresh or robot_id not in self._noise:
            self._noise[robot_id] = self._rng.normal(0.0, self.std, size=action.shape).astype(np.float32)
        noisy = action + self._noise[robot_id]
        if bounds is not None:
            noisy = np.clip(noisy, bounds[0], bounds[1])
        noisy = noisy.astype(np.float32)
        return noisy, True
