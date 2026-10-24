"""Robots must not get stuck in endless episodes.

Robots stand back up where they fall. One that drifted to a wall stood up
against it, leaned on it above the fall height and never ended its episode,
filling fleet experience with 1000s of ticks of slumping. Robots now end
episodes at the test track's cap, and one that ends near a wall
respawns at its spawn point.
"""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

from locomotion_reward import MAX_EPISODE_STEPS  # noqa: E402
from mujoco_env import NUM_ACTUATORS, LabSimulation  # noqa: E402
from transitions import TransitionRecorder  # noqa: E402

def test_episode_truncates_at_test_track_cap_without_counting_a_fall():
    sim = LabSimulation()
    robot = sim.robots[0]
    robot["episode_steps"] = MAX_EPISODE_STEPS - 1
    episodes, falls = robot["episode_count"], robot["fall_count"]

    sim.step({robot["id"]: np.zeros(NUM_ACTUATORS)})

    assert robot["episode_count"] == episodes + 1
    assert robot["truncation_count"] == 1
    assert robot["fall_count"] == falls
    assert robot["episode_steps"] == 0


def test_reset_near_a_wall_respawns_at_the_robots_spawn_point():
    sim = LabSimulation()
    sim.spawn_robot()  # robot 2 spawns away from the origin
    robot = sim.robots[1]
    qs = robot["qpos_start"]
    spawn = sim.model.qpos0[qs:qs + 2].copy()
    assert np.linalg.norm(spawn) > 1.0
    sim.data.qpos[qs:qs + 2] = (4.8, -4.9)

    sim._reset_robot(robot)

    np.testing.assert_allclose(sim.data.qpos[qs:qs + 2], spawn)


def test_reset_keeps_a_robot_where_it_fell_in_open_floor():
    sim = LabSimulation()
    robot = sim.robots[0]
    qs = robot["qpos_start"]
    sim.data.qpos[qs:qs + 2] = (1.5, -2.0)

    sim._reset_robot(robot)

    np.testing.assert_allclose(sim.data.qpos[qs:qs + 2], (1.5, -2.0))


def test_truncation_starts_a_new_episode_without_a_terminal_record():
    rec = TransitionRecorder("r1")
    rec.before_step([0.0] * 86, [0.0] * 25, False, cum_reward=0.0, record=True)
    old_episode = rec.episode_id

    out = rec.after_step(cum_reward=1.0, fell=False, truncated=True)

    assert out == [], "a time-limit end is not a terminal state for offline RL"
    assert rec.episode_id != old_episode
    nxt = rec.before_step([0.0] * 86, [0.0] * 25, False, cum_reward=1.0, record=True)
    assert nxt == [], "the dropped transition must not be closed into the new episode"


def test_respawn_is_not_scored_as_walking_backwards():
    sim = LabSimulation()
    robot = sim.robots[0]
    qs = robot["qpos_start"]
    sim.data.qpos[qs:qs + 2] = (4.8, 0.0)
    robot["prev_x"] = 4.8

    sim._reset_robot(robot)
    robot["cum_reward"] = 0.0
    sim.step({robot["id"]: np.zeros(NUM_ACTUATORS)}, task_commands={robot["id"]: "walk"})

    assert robot["cum_reward"] > -1.0, "the teleport back to spawn was credited as velocity"
