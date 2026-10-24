"""The test track's scenario presets must change the physics they name.

evaluate_policy.py listed gravity/friction/wind presets but never applied
them: every "low_friction" or "windy" scenario ran standard physics, so the
pass rate the promotion gate trusts was five copies of one test.
"""

import numpy as np
import pytest

import evaluate_policy as ep
from fleetos_env import FleetOSHumanoidEnv
from observation import NUM_ACTUATORS


@pytest.fixture
def env():
    return FleetOSHumanoidEnv()


def test_standard_conditions_are_the_scene_defaults(env):
    gravity = env.model.opt.gravity.copy()
    friction = env.model.geom_friction.copy()
    env.set_conditions(**ep.SCENARIO_PRESETS["standard"])
    env.reset(seed=0)
    np.testing.assert_allclose(env.model.opt.gravity, gravity)
    np.testing.assert_allclose(env.model.geom_friction, friction)
    assert not env.data.xfrc_applied.any()


def test_gravity_and_friction_are_applied(env):
    base = env.model.geom_friction[:, 0].copy()
    env.set_conditions(gravity=-11.0, friction=0.5, wind=0.0)
    np.testing.assert_allclose(env.model.opt.gravity, (0.0, 0.0, -11.0))
    np.testing.assert_allclose(env.model.geom_friction[:, 0], base * 0.5)
    # Torsional/rolling friction untouched; scaling is relative to the scene, not compounding.
    env.set_conditions(gravity=-9.81, friction=1.0, wind=0.0)
    np.testing.assert_allclose(env.model.geom_friction[:, 0], base)


def test_wind_pushes_the_robot_sideways_and_survives_reset(env):
    def lateral_drift(wind):
        env.set_conditions(gravity=-9.81, friction=1.0, wind=wind)
        env.reset(seed=0)
        y0 = env.data.qpos[env._root_qpos_adr + 1]
        for _ in range(20):
            env.step(np.zeros(NUM_ACTUATORS, dtype=np.float32))
        return env.data.qpos[env._root_qpos_adr + 1] - y0

    assert lateral_drift(ep.SCENARIO_PRESETS["windy"]["wind"]) > lateral_drift(0.0) + 1e-3


class _RecordingVecEnv:
    """Stands in for the evaluator's (possibly normalized) DummyVecEnv."""

    def __init__(self):
        self.calls = []

    def env_method(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def reset(self):
        return np.zeros((1, 86), dtype=np.float32)

    def step(self, action):
        return np.zeros((1, 86), dtype=np.float32), np.array([1.0]), np.array([True]), [{}]


class _ZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, NUM_ACTUATORS), dtype=np.float32), None


@pytest.mark.parametrize("preset", list(ep.SCENARIO_PRESETS))
def test_each_scenario_runs_under_its_preset(preset):
    env = _RecordingVecEnv()
    ep.run_scenario(_ZeroPolicy(), env, f"{preset}-000", preset, ep.SCENARIO_PRESETS[preset])
    assert env.calls == [("set_conditions", ep.SCENARIO_PRESETS[preset])]
