"""Reward sanity for fleet data: spawning must not create fake forward velocity."""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

from mujoco_env import LabSimulation  # noqa: E402

WALK_STEP_REWARD_BOUND = 20.0  # a humanoid cannot gain more than this per step


def test_first_walk_step_after_spawn_has_no_reward_spike():
    sim = LabSimulation()
    sim.spawn_robot()  # second robot spawns away from x=0 and reloads the model
    for robot in sim.robots:
        assert robot["prev_x"] == float(sim.data.qpos[robot["qpos_start"]])

    before = {r["id"]: r["cum_reward"] for r in sim.robots}
    actions = {r["id"]: np.zeros(25, dtype=np.float32) for r in sim.robots}
    sim.step(actions, task_commands={rid: "walk" for rid in actions})
    for r in sim.robots:
        assert abs(r["cum_reward"] - before[r["id"]]) < WALK_STEP_REWARD_BOUND


def test_crouching_robot_is_penalised_like_training():
    # Training (locomotion_reward.FALL_Z) treats a crouched torso as a fall; robots must too.
    sim = LabSimulation()
    robot = sim.robots[0]
    robot["prev_x"] = float(sim.data.qpos[robot["qpos_start"]])
    reward = sim._compute_reward(robot, "walk", height=0.57, x_pos=robot["prev_x"],
                                 dt=0.021, ctrl_cost=0.0)
    assert reward == -10.0, "simulator rewards a crouch the training env counts as a fall"
