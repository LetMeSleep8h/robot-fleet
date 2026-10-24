"""Robots must accept the full per-joint action range policies are trained with.

Training clips each action to its actuator's ctrlrange (e.g. knee down to
-2.8 rad); robots used to clip everything to [-1, 1], cutting large joint
targets short.
"""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

from edge_policy import EdgePolicyManager  # noqa: E402
from exploration import Explorer  # noqa: E402
from mujoco_env import NUM_ACTUATORS, LabSimulation  # noqa: E402


def _robot_bounds(sim, robot):
    start = robot["actuator_start"]
    ranges = sim.model.actuator_ctrlrange[start:start + NUM_ACTUATORS]
    return ranges[:, 0], ranges[:, 1]


def test_step_clips_to_each_actuators_control_range():
    sim = LabSimulation()
    robot = sim.robots[0]
    low, high = _robot_bounds(sim, robot)
    assert low.min() < -1.0, "scene should have joints whose range exceeds [-1, 1]"

    start = robot["actuator_start"]
    sim.step({robot["id"]: np.full(NUM_ACTUATORS, -5.0)})
    np.testing.assert_allclose(sim.data.ctrl[start:start + NUM_ACTUATORS], low, rtol=0, atol=1e-6)

    sim.step({robot["id"]: np.full(NUM_ACTUATORS, 5.0)})
    np.testing.assert_allclose(sim.data.ctrl[start:start + NUM_ACTUATORS], high, rtol=0, atol=1e-6)


def test_action_bounds_helper_matches_scene():
    sim = LabSimulation()
    robot = sim.robots[0]
    low, high = sim.action_bounds(robot["id"])
    exp_low, exp_high = _robot_bounds(sim, robot)
    np.testing.assert_array_equal(low, exp_low)
    np.testing.assert_array_equal(high, exp_high)


def test_exploration_noise_respects_given_bounds_not_unit_box():
    explorer = Explorer(fraction=1.0, std=0.5, rng=np.random.default_rng(0))
    low = np.full(NUM_ACTUATORS, -2.8, dtype=np.float32)
    high = np.full(NUM_ACTUATORS, 0.5, dtype=np.float32)
    action = np.full(NUM_ACTUATORS, -2.5, dtype=np.float32)

    noisy, explored = explorer.apply("robot-x", action, refresh=True, bounds=(low, high))

    assert explored
    assert noisy.min() >= -2.8 and noisy.max() <= 0.5
    assert noisy.min() < -1.0, "noise must not be clipped back into [-1, 1]"


class _WideRangePolicy:
    """A policy whose (trained) actions lie outside [-1, 1]."""

    def predict(self, obs, deterministic=False):
        return np.full(NUM_ACTUATORS, -2.5, dtype=np.float32), None


def test_edge_action_is_not_clipped_to_unit_box():
    def http_get(url, headers, timeout):
        return 200, {"artifact_url": "training/x/policy.zip", "model_id": "m", "source": "assigned"}

    mgr = EdgePolicyManager(api_url="http://unused", api_key="k",
                            fetch_and_load=lambda url: _WideRangePolicy(), http_get=http_get)
    mgr.register_robot("r1")
    mgr.refresh_once()

    act = mgr.action("r1", np.zeros(86, dtype=np.float32))

    assert act is not None and np.allclose(act, -2.5), "edge must pass the policy's action through; physics clips it"
