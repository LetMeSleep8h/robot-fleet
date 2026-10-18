"""
Per-robot transition recording for offline RL from fleet telemetry.

A transition opens on a record tick with the observation the policy saw and the
action actually applied, and closes on the next record tick (or on a fall).
The closed record carries the reward earned by that action in between, so the
training side pairs records as (obs, action, reward, next record's obs).

The simulator resets a fallen robot inside the physics step, so falls are
detected by the caller (episode counter change) and reported via after_step,
which emits a terminal record immediately.
"""

import uuid


def _new_episode_id() -> str:
    return uuid.uuid4().hex


class TransitionRecorder:
    """Tracks one robot's episode id, step index and open transition."""

    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.episode_id = _new_episode_id()
        self._next_step = 0
        self._pending: dict | None = None

    def before_step(self, observation, action, exploration: bool,
                    cum_reward: float, record: bool) -> list[dict]:
        """Call before the physics step. On record ticks, closes the open
        transition (returned) and opens a new one for this tick's (obs, action)."""
        if not record:
            return []
        out = []
        if self._pending is not None:
            out.append(self._close(cum_reward, done=False))
        self._pending = {
            "observation": [float(x) for x in observation],
            "action_taken": [float(x) for x in action],
            "exploration": bool(exploration),
            "step": self._next_step,
            "cum_start": float(cum_reward),
        }
        self._next_step += 1
        return out

    def after_step(self, cum_reward: float, fell: bool) -> list[dict]:
        """Call after the physics step. A fall closes the open transition as
        terminal and starts a new episode."""
        if not fell:
            return []
        out = []
        if self._pending is not None:
            out.append(self._close(cum_reward, done=True))
        self.episode_id = _new_episode_id()
        self._next_step = 0
        self._pending = None
        return out

    def _close(self, cum_reward: float, done: bool) -> dict:
        p = self._pending
        self._pending = None
        return {
            "observation": p["observation"],
            "action_taken": p["action_taken"],
            "exploration": p["exploration"],
            "step": p["step"],
            "episode_id": self.episode_id,
            "step_reward": float(cum_reward) - p["cum_start"],
            "episode_done": done,
        }
