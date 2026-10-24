"""Observations must be egocentric: the same robot state gives the same
observation wherever the robot stands and whichever way it faces.

World-frame hand/object positions made policies depend on the robot being at
the origin with objects in their start spots; fleet robots spawn elsewhere,
walk away and knock objects around.
"""

import math

import mujoco
import numpy as np
import pytest

import observation as ob
from fleetos_env import FleetOSHumanoidEnv


def _yaw_quat(yaw: float) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion for a rotation about +z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _rot_z(yaw: float, v: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _object_slots(model):
    return {name: (model.jnt_qposadr[model.body_jntadr[model.body(name).id]],
                   model.jnt_dofadr[model.body_jntadr[model.body(name).id]])
            for name in ob.OBJECT_NAMES}


def _place(env, x, y, yaw, obj_offsets, obj_vel_world=None):
    """Put the robot at (x, y) facing yaw and each object at a robot-relative offset."""
    env.reset(seed=0)
    m, d = env.model, env.data
    qs, ds = env._root_qpos_adr, env._root_dof_adr
    d.qpos[qs:qs + 2] = (x, y)
    d.qpos[qs + 3:qs + 7] = _yaw_quat(yaw)
    d.qvel[:] = 0.0
    for name, (qa, va) in _object_slots(m).items():
        off = _rot_z(yaw, np.asarray(obj_offsets[name], dtype=float))
        d.qpos[qa:qa + 3] = (x + off[0], y + off[1], off[2])
        d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
        if obj_vel_world is not None:
            d.qvel[va:va + 3] = obj_vel_world[name]
    mujoco.mj_forward(m, d)
    return ob.build_observation_from_mujoco(m, d, qs, ds, task_command="walk")


OFFSETS = {"red_box": (1.0, 0.5, 0.4), "blue_cylinder": (-0.8, 1.2, 0.4)}


@pytest.fixture(scope="module")
def env():
    return FleetOSHumanoidEnv()


def test_shape_and_version():
    assert ob.OBS_DIM == 86
    assert ob.OBS_VERSION == 3


def test_translation_invariant(env):
    at_origin = _place(env, 0.0, 0.0, 0.0, OFFSETS)
    elsewhere = _place(env, 3.7, -2.4, 0.0, OFFSETS)
    np.testing.assert_allclose(elsewhere, at_origin, atol=1e-4)


def test_heading_invariant(env):
    facing_x = _place(env, 1.0, 1.0, 0.0, OFFSETS)
    turned = _place(env, 1.0, 1.0, 1.1, OFFSETS)
    np.testing.assert_allclose(turned, facing_x, atol=1e-4)


def test_hand_positions_are_body_relative(env):
    obs = _place(env, 5.0, 5.0, 0.0, OFFSETS)
    hands = obs[ob.HAND_SLICE]
    assert np.all(np.abs(hands) < 1.5), f"hand positions look world-frame: {hands}"


def test_object_velocity_is_linear_and_heading_relative(env):
    yaw = math.pi / 2  # robot faces +y
    vel = {"red_box": np.array([0.0, 2.0, 0.0]), "blue_cylinder": np.zeros(3)}
    obs = _place(env, 0.0, 0.0, yaw, OFFSETS, obj_vel_world=vel)
    red_vel = obs[ob.OBJECT_VEL_SLICE][:3]
    # Moving along the robot's facing direction reads as +forward (x) in its frame.
    np.testing.assert_allclose(red_vel, [2.0, 0.0, 0.0], atol=1e-3)
