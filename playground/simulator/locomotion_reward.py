"""Locomotion reward and fall rule shared by training and the robots.

Byte-identical copies live in platform/training (FleetOS-Humanoid-v2, the
training env and test track) and playground/simulator (the robots);
platform/training/test_parity.py fails CI if they drift. A policy is only
judged fairly if robots score and terminate exactly as it was trained.
"""

import numpy as np

FALL_Z = 0.3            # torso height below which the robot has fallen (episode ends)
FALL_PENALTY = -10.0
ALIVE_BONUS = 1.0
CTRL_COST_WEIGHT = 0.1


def has_fallen(height: float) -> bool:
    """True when the torso is below FALL_Z."""
    return height < FALL_Z


def control_cost(action) -> float:
    """Quadratic effort penalty for the applied actuator commands."""
    return CTRL_COST_WEIGHT * float(np.sum(np.square(action)))


def locomotion_reward(forward_velocity: float, height: float, ctrl_cost: float) -> float:
    """Per-control-tick reward for walking: progress + staying up - effort."""
    if has_fallen(height):
        return FALL_PENALTY
    return forward_velocity + ALIVE_BONUS - ctrl_cost
