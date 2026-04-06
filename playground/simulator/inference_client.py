"""
Async inference client for the simulator.

Calls the platform inference server's /predict endpoint in a background
thread. The simulator's 50 Hz physics loop never blocks — it always
gets the latest cached action.

Effective inference rate: ~7-10 Hz (inference takes ~100-150ms).
Between predictions, the last action is held (position actuators
maintain the joint angles via PD control).
"""

import logging
import threading
import time

import numpy as np
import requests

log = logging.getLogger(__name__)

NUM_ACTUATORS = 25


class InferenceClient:
    """Non-blocking inference client with action caching."""

    def __init__(self, endpoint: str = "http://inference:8081"):
        self._endpoint = endpoint.rstrip("/")
        self._cached_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        self._pending = False
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._min_interval = 0.1  # max 10 Hz inference rate
        self._consecutive_errors = 0
        self._max_errors_before_backoff = 5

    def request_action(self, observation: list, task_type: str, goal: list | None = None):
        """Start inference in background (non-blocking).

        If a request is already in flight, this is a no-op.
        Rate-limited to ~10 Hz to avoid overwhelming the server.
        """
        now = time.monotonic()
        if self._pending or (now - self._last_request_time) < self._min_interval:
            return

        # Back off if inference server is consistently failing
        if self._consecutive_errors >= self._max_errors_before_backoff:
            if (now - self._last_request_time) < 5.0:
                return
            self._consecutive_errors = 0  # retry after backoff

        self._pending = True
        self._last_request_time = now
        threading.Thread(
            target=self._call_predict,
            args=(observation, task_type, goal),
            daemon=True,
        ).start()

    def get_cached_action(self) -> np.ndarray:
        """Return the latest predicted action (always non-blocking)."""
        with self._lock:
            return self._cached_action.copy()

    def _call_predict(self, observation: list, task_type: str, goal: list | None):
        """Background thread: call /predict and update cached action."""
        try:
            payload = {
                "observation": observation,
                "instruction": task_type,
            }
            if goal:
                payload["goal"] = goal

            resp = requests.post(
                f"{self._endpoint}/predict",
                json=payload,
                timeout=2,
            )

            if resp.ok:
                data = resp.json()

                # Prefer flat "action" array (new format)
                if "action" in data and isinstance(data["action"], list):
                    action = np.array(data["action"][:NUM_ACTUATORS], dtype=np.float32)
                    if len(action) < NUM_ACTUATORS:
                        action = np.pad(action, (0, NUM_ACTUATORS - len(action)))
                # Fall back to predicted_actions dict format (legacy)
                elif "predicted_actions" in data:
                    action = self._parse_predicted_actions(data["predicted_actions"])
                else:
                    return

                with self._lock:
                    self._cached_action = action
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1
                log.debug("Inference returned %d", resp.status_code)

        except requests.RequestException:
            self._consecutive_errors += 1
        except Exception as e:
            self._consecutive_errors += 1
            log.debug("Inference error: %s", e)
        finally:
            self._pending = False

    def _parse_predicted_actions(self, predicted_actions: list) -> np.ndarray:
        """Parse the legacy predicted_actions format (list of joint dicts)."""
        from joint_mapping import FLEET_TO_MUJOCO_ACTUATOR
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        for entry in predicted_actions:
            joint_name = entry.get("joint", "")
            torque = entry.get("torque", entry.get("position", 0.0))
            if joint_name in FLEET_TO_MUJOCO_ACTUATOR:
                idx = FLEET_TO_MUJOCO_ACTUATOR[joint_name]
                action[idx] = np.clip(torque, -1.0, 1.0)
        return action
