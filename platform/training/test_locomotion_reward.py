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


CROUCH_HEIGHT = 0.57    # torso height of the crawl gait a 0.3 m fall rule let PPO learn
STANDING_HEIGHT = 1.22  # spawn torso height (feet on the floor)


def test_crouching_counts_as_a_fall():
    # With a 0.3 m rule PPO learned to crawl at torso 0.57 m: alive bonus without
    # balancing. A crouch must end the episode like a fall.
    assert lr.has_fallen(CROUCH_HEIGHT)
    assert lr.locomotion_reward(1.0, CROUCH_HEIGHT, 0.0) == lr.FALL_PENALTY


def test_standing_robot_has_not_fallen():
    assert not lr.has_fallen(STANDING_HEIGHT)
    assert lr.FALL_Z < STANDING_HEIGHT, "robots would count as fallen the moment they spawn"
