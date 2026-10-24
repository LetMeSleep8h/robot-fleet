"""Tests for evaluate_policy.py's model loading (online PPO and offline SAC models)."""

from unittest import mock

import pytest

import evaluate_policy as ep


def test_loads_ppo_models():
    sentinel = object()
    with mock.patch.object(ep.PPO, "load", return_value=sentinel) as ppo, \
         mock.patch.object(ep.SAC, "load") as sac:
        model, algo = ep.load_policy("policy.zip")
    assert model is sentinel and algo == "PPO"
    ppo.assert_called_once_with("policy.zip")
    sac.assert_not_called()


def test_falls_back_to_sac_for_offline_models():
    sentinel = object()
    with mock.patch.object(ep.PPO, "load", side_effect=ValueError("not a PPO zip")), \
         mock.patch.object(ep.SAC, "load", return_value=sentinel):
        model, algo = ep.load_policy("policy.zip")
    assert model is sentinel and algo == "SAC"


def test_unloadable_model_raises_with_all_reasons():
    with mock.patch.object(ep.PPO, "load", side_effect=ValueError("ppo says no")), \
         mock.patch.object(ep.SAC, "load", side_effect=ValueError("sac says no")):
        with pytest.raises(RuntimeError, match="ppo says no.*sac says no"):
            ep.load_policy("policy.zip")


def test_real_offline_policy_round_trip(tmp_path):
    """A model saved by train_offline loads and acts through the evaluator's loader."""
    import numpy as np

    import train_offline as to

    args = to.build_parser().parse_args(["--job-id", "j", "--dataset-manifest", "m", "--batch-size", "4"])
    model = to.make_model(args, buffer_size=8)
    model.save(str(tmp_path / "policy"))

    loaded, algo = ep.load_policy(str(tmp_path / "policy.zip"))
    assert algo == "SAC"
    action, _ = loaded.predict(np.zeros(to.OBS_DIM, dtype=np.float32), deterministic=True)
    assert action.shape == (to.ACT_DIM,)


def test_refuses_policies_without_matching_obs_version(tmp_path):
    import policy_meta as pm
    model = tmp_path / "policy.zip"
    model.write_bytes(b"zip")
    with pytest.raises(ep.IncompatiblePolicy, match="policy_meta.json"):
        ep.check_compatible(str(model))
    pm.write_policy_meta(tmp_path)
    ep.check_compatible(str(model))  # compatible: no exception


# issue #13: which gate failed, and the episode-length gate has its own reason.

class _FakeEnv:
    """Records steps and yields a scripted reward, then ends the episode."""

    def __init__(self, reward, steps):
        self.reward = reward
        self.steps = steps
        self._n = 0

    def env_method(self, *a, **k):
        pass

    def reset(self):
        return None

    def step(self, action):
        self._n += 1
        return None, [self.reward], self._n >= self.steps, {}


def _run(monkeypatch, reward, steps):
    env = _FakeEnv(reward, steps)
    model = type("M", (), {"predict": staticmethod(lambda obs, deterministic=True: (None, None))})
    result = ep.run_scenario(model, env, "s1", "preset", {})
    return result


def test_low_reward_failure_names_the_reward_gate(monkeypatch):
    result = _run(monkeypatch, reward=1.0, steps=400)
    assert not result["passed"]
    assert "reward" in result["failure_reason"]
    assert "500" in result["failure_reason"]


def test_short_high_reward_failure_names_the_length_gate(monkeypatch):
    result = _run(monkeypatch, reward=600.0, steps=50)
    assert not result["passed"]
    assert "steps" in result["failure_reason"]
    assert "reward" not in result["failure_reason"]


def test_passing_scenario_has_no_reason(monkeypatch):
    result = _run(monkeypatch, reward=600.0, steps=120)
    assert result["passed"] and result["failure_reason"] == ""
