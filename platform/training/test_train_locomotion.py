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
