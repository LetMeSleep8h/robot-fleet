"""Tests for the locomotion reward shared by training and the simulator."""

import numpy as np

import locomotion_reward as lr


def test_fall_is_penalised_and_terminal():
    assert lr.has_fallen(lr.FALL_Z - 0.01)
    assert not lr.has_fallen(lr.FALL_Z + 0.01)
    assert lr.locomotion_reward(5.0, lr.FALL_Z - 0.01, 0.0) == lr.FALL_PENALTY


def test_upright_reward_is_progress_plus_alive_minus_effort():
    action = np.full(25, 0.5)
    cost = lr.control_cost(action)
    assert cost == lr.CTRL_COST_WEIGHT * float(np.sum(action ** 2))
    assert lr.locomotion_reward(1.2, 1.1, cost) == 1.2 + lr.ALIVE_BONUS - cost


def test_low_but_upright_robot_is_not_penalised():
    # Between the old simulator threshold (0.4) and FALL_Z the robot hasn't fallen.
    assert lr.locomotion_reward(0.0, 0.35, 0.0) == lr.ALIVE_BONUS
