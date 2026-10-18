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
