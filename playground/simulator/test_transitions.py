"""Tests for exploration noise and per-robot transition recording."""

import numpy as np
import pytest

from exploration import Explorer, is_exploration_robot
from transitions import TransitionRecorder

OBS = [0.5] * 86
ACT = [0.1] * 25


# --- exploration -----------------------------------------------------------

def test_exploration_robot_selection_is_deterministic():
    picks = [is_exploration_robot(f"robot-{i:04d}", 0.2) for i in range(1000)]
    assert picks == [is_exploration_robot(f"robot-{i:04d}", 0.2) for i in range(1000)]
    assert 150 < sum(picks) < 250  # ~20% of the fleet


@pytest.mark.parametrize("fraction,expected", [(0.0, False), (1.0, True)])
def test_exploration_fraction_bounds(fraction, expected):
    assert all(is_exploration_robot(f"r{i}", fraction) == expected for i in range(50))


def test_explorer_adds_bounded_noise_only_for_exploration_robots():
    ex = Explorer(fraction=1.0, std=0.5, rng=np.random.default_rng(0))
    action = np.full(25, 0.9, dtype=np.float32)
    bounds = (np.full(25, -1.0, dtype=np.float32), np.full(25, 1.0, dtype=np.float32))
    noisy, explored = ex.apply("robot-0001", action, refresh=True, bounds=bounds)
    assert explored
    assert noisy.shape == (25,)
    assert not np.allclose(noisy, action)
    assert noisy.min() >= -1.0 and noisy.max() <= 1.0  # clipped to the given bounds

    calm = Explorer(fraction=0.0, std=0.5, rng=np.random.default_rng(0))
    same, explored = calm.apply("robot-0001", action, refresh=True)
    assert not explored
    np.testing.assert_array_equal(same, action)


def test_explorer_holds_noise_until_refresh():
    ex = Explorer(fraction=1.0, std=0.3, rng=np.random.default_rng(1))
    base = np.zeros(25, dtype=np.float32)
    first, _ = ex.apply("r", base, refresh=True)
    held, _ = ex.apply("r", base, refresh=False)
    np.testing.assert_array_equal(first, held)  # constant over a transition interval
    fresh, _ = ex.apply("r", base, refresh=True)
    assert not np.allclose(first, fresh)


# --- transition recording ----------------------------------------------------

def test_record_carries_reward_earned_until_next_record():
    rec = TransitionRecorder("robot-0001")
    assert rec.before_step(OBS, ACT, exploration=False, cum_reward=0.0, record=True) == []
    rec.after_step(cum_reward=1.0, fell=False)
    out = rec.before_step([0.6] * 86, ACT, exploration=False, cum_reward=2.5, record=True)

    assert len(out) == 1
    r = out[0]
    assert r["observation"] == OBS and r["action_taken"] == ACT
    assert r["step_reward"] == pytest.approx(2.5)  # everything since the record opened
    assert r["step"] == 0 and r["episode_done"] is False
    assert r["episode_id"] == rec.episode_id


def test_steps_increment_within_an_episode():
    rec = TransitionRecorder("r")
    steps = []
    for i in range(4):
        steps += [r["step"] for r in rec.before_step(OBS, ACT, False, float(i), record=True)]
        rec.after_step(float(i) + 0.5, fell=False)
    assert steps == [0, 1, 2]


def test_fall_emits_terminal_record_and_starts_new_episode():
    rec = TransitionRecorder("r")
    rec.before_step(OBS, ACT, exploration=True, cum_reward=10.0, record=True)
    first_episode = rec.episode_id
    out = rec.after_step(cum_reward=7.0, fell=True)

    assert len(out) == 1
    t = out[0]
    assert t["episode_done"] is True
    assert t["step_reward"] == pytest.approx(-3.0)
    assert t["episode_id"] == first_episode and t["exploration"] is True
    assert rec.episode_id != first_episode

    nxt = rec.before_step(OBS, ACT, False, 7.0, record=True)
    assert nxt == []  # nothing pending from the finished episode
    rec.after_step(8.0, fell=False)
    again = rec.before_step(OBS, ACT, False, 8.0, record=True)
    assert again[0]["step"] == 0 and again[0]["episode_id"] == rec.episode_id


def test_non_record_ticks_do_not_open_or_close_transitions():
    rec = TransitionRecorder("r")
    rec.before_step(OBS, ACT, False, 0.0, record=True)
    for t in range(4):
        assert rec.before_step([9.0] * 86, ACT, False, float(t), record=False) == []
        assert rec.after_step(float(t) + 1, fell=False) == []
    out = rec.before_step(OBS, ACT, False, 4.0, record=True)
    assert out[0]["observation"] == OBS  # the obs from when the action was chosen
