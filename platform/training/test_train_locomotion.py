"""Tests for train_locomotion.py's tunable hyperparameters and Katib metric output."""

import re
from unittest import mock

import train_locomotion as tl

# Katib's default StdOut metrics-collector filter.
KATIB_METRIC_RE = re.compile(r"([\w|-]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?)")


def test_hyperparameter_defaults_match_previous_hardcoded_values():
    args = tl.build_parser().parse_args(["--job-id", "j1"])
    assert tl.hyperparameters(args) == {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "ent_coef": 0.0,
    }


def test_hyperparameters_are_tunable_from_cli():
    args = tl.build_parser().parse_args([
        "--job-id", "j1", "--learning-rate", "0.001", "--n-steps", "512",
        "--batch-size", "128", "--gamma", "0.95", "--ent-coef", "0.01",
    ])
    hp = tl.hyperparameters(args)
    assert hp == {"learning_rate": 0.001, "n_steps": 512, "batch_size": 128, "gamma": 0.95, "ent_coef": 0.01}


def test_create_model_uses_given_hyperparameters():
    hp = {"learning_rate": 0.001, "n_steps": 512, "batch_size": 128, "gamma": 0.95, "ent_coef": 0.01}
    with mock.patch.object(tl, "PPO") as ppo:
        tl._create_model(env=object(), device="cpu", hp=hp)
    kwargs = ppo.call_args.kwargs
    for key, value in hp.items():
        assert kwargs[key] == value


def test_katib_metric_line_is_parseable():
    line = tl.katib_metric_line(12.3456)
    matches = {m.group(1): float(m.group(2)) for m in KATIB_METRIC_RE.finditer(line)}
    assert matches == {"eval_mean_reward": 12.35}


def test_default_env_id_is_registered():
    import gymnasium as gym
    default_env = tl.build_parser().parse_args(["--job-id", "j1"]).env_id
    assert default_env in gym.registry, f"default --env-id {default_env!r} is not a registered environment"


# --- parallel environments -------------------------------------------------------

def test_single_env_runs_in_process():
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = tl.build_vec_env("FleetOS-Humanoid-v2", n_envs=1)
    try:
        assert isinstance(env, DummyVecEnv) and env.num_envs == 1
    finally:
        env.close()


def test_several_envs_run_in_parallel_processes():
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = tl.build_vec_env("FleetOS-Humanoid-v2", n_envs=2)
    try:
        assert isinstance(env, SubprocVecEnv) and env.num_envs == 2
        obs = env.reset()
        assert obs.shape == (2, 86)
        # Each worker is seeded differently, so episodes don't run in lockstep.
        assert not (obs[0] == obs[1]).all()
    finally:
        env.close()


def test_n_envs_flag_defaults_to_one():
    assert tl.build_parser().parse_args(["--job-id", "j1"]).n_envs == 1
    assert tl.build_parser().parse_args(["--job-id", "j1", "--n-envs", "8"]).n_envs == 8


# --- final evaluation --------------------------------------------------------------

def test_eval_env_scores_raw_reward_with_frozen_training_stats():
    from stable_baselines3.common.vec_env import VecNormalize
    train_env = VecNormalize(tl.build_vec_env("FleetOS-Humanoid-v2", n_envs=1), norm_obs=True, norm_reward=True)
    train_env.obs_rms.mean[:] = 3.0
    try:
        eval_env = tl.make_eval_env("FleetOS-Humanoid-v2", train_env)
        assert eval_env.norm_reward is False, "eval reward must be the test track's raw reward"
        assert eval_env.training is False, "evaluation must not update normalization stats"
        assert (eval_env.obs_rms.mean == 3.0).all()
        eval_env.obs_rms.mean[:] = 0.0
        assert (train_env.obs_rms.mean == 3.0).all(), "eval env must not share mutable stats"
        eval_env.close()
    finally:
        train_env.close()


# --- training metrics ----------------------------------------------------------------

def test_vec_env_reports_finished_episodes():
    import numpy as np
    env = tl.build_vec_env("FleetOS-Humanoid-v2", n_envs=1)
    try:
        env.reset()
        for _ in range(1000):
            _, _, dones, infos = env.step(np.zeros((1, 25), dtype=np.float32))
            if dones[0]:
                break
        assert dones[0], "a passive humanoid should fall within 1000 ticks"
        assert {"r", "l"} <= set(infos[0]["episode"]), "episodes need Monitor stats for training metrics"
    finally:
        env.close()


class _Model:
    def __init__(self):
        self.num_timesteps = 0


def _drive(callback, steps):
    """steps: list of per-env info lists, one per callback step."""
    callback.model = _Model()
    for infos in steps:
        callback.model.num_timesteps += len(infos)
        callback.num_timesteps = callback.model.num_timesteps
        callback.locals = {"infos": infos}
        callback._on_step()


def test_metrics_record_each_finished_episode_once(tmp_path):
    cb = tl.MetricsCallback(str(tmp_path / "metrics.json"))
    _drive(cb, [
        [{}, {}],
        [{"episode": {"r": 10.0, "l": 30}}, {}],
        [{}, {}],
        [{}, {"episode": {"r": 20.0, "l": 40}}],
        [{}, {}],
    ])
    assert cb.metrics["episode_rewards"] == [10.0, 20.0]
    assert cb.metrics["episode_lengths"] == [30, 40]


def test_metrics_keep_a_learning_curve(tmp_path):
    import json
    cb = tl.MetricsCallback(str(tmp_path / "metrics.json"), flush_every=2)
    _drive(cb, [[{"episode": {"r": float(i), "l": 10}}] for i in range(4)])
    cb._on_training_end()
    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert summary["total_episodes"] == 4
    curve = summary["history"]
    assert [p["timesteps"] for p in curve][:2] == [2, 4]
    assert curve[-1]["mean_reward"] == 1.5 and "elapsed_seconds" in curve[-1]
