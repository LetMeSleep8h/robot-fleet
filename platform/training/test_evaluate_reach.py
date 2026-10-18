"""Tests for evaluate_reach.py's results contract (shared with evaluate_policy.py)."""

import json
from pathlib import Path
from unittest import mock

import pytest

import evaluate_reach as er


@pytest.mark.parametrize("model_path,expected", [
    ("s3://fleetos-models/training/j1/policy.zip", "training/j1/policy.zip"),
    ("training/j1/policy.zip", "training/j1/policy.zip"),
    ("/tmp/local/policy.zip", None),
    ("s3://other-bucket/training/j1/policy.zip", None),
])
def test_s3_key_from_model_path(model_path, expected):
    assert er.s3_key_from_model_path(model_path, "fleetos-models") == expected


def test_build_results_has_contract_keys():
    stats = {"close_range": {"success_rate": 0.5}}
    r = er.build_results("e1", "m", stats, passed=3, total=4,
                         rewards=[-10.0, -20.0], pass_threshold=0.5)
    assert r["eval_id"] == "e1"
    assert r["status"] == "completed"
    assert r["scenarios_total"] == 4
    assert r["scenarios_passed"] == 3
    assert r["pass_rate"] == 0.75
    assert r["mean_reward"] == -15.0
    assert r["results"] == stats
    assert r["pass"] is True


def test_build_results_empty_is_zero():
    r = er.build_results("e1", "m", {}, passed=0, total=0, rewards=[], pass_threshold=0.5)
    assert r["pass_rate"] == 0.0
    assert r["mean_reward"] == 0.0
    assert r["pass"] is False


def test_upload_results_writes_evaluations_key_and_skips_dirs(tmp_path: Path):
    (tmp_path / "eval_results.json").write_text("{}")
    (tmp_path / "model").mkdir()
    args = er.parse_args(["--eval-id", "e9", "--model-path", "x", "--s3-endpoint", "minio:9000"])
    client = mock.Mock()
    with mock.patch.object(er, "_s3_client", return_value=client):
        er.upload_results(args, tmp_path)
    client.fput_object.assert_called_once_with(
        "fleetos-models", "evaluations/e9/eval_results.json", str(tmp_path / "eval_results.json"))


def test_upload_results_failure_is_fatal(tmp_path: Path):
    (tmp_path / "eval_results.json").write_text("{}")
    args = er.parse_args(["--eval-id", "e9", "--model-path", "x", "--s3-endpoint", "minio:9000"])
    client = mock.Mock()
    client.fput_object.side_effect = RuntimeError("s3 down")
    with mock.patch.object(er, "_s3_client", return_value=client):
        with pytest.raises(RuntimeError):
            er.upload_results(args, tmp_path)


def test_scenarios_total_sets_episodes_per_scenario():
    args = er.parse_args(["--model-path", "x", "--scenarios", "100"])
    assert args.episodes == 100 // len(er.SCENARIOS)
    args = er.parse_args(["--model-path", "x", "--scenarios", "2"])
    assert args.episodes == 1


def test_notify_callback_payload_shape():
    results = er.build_results("e1", "m", {"a": {}}, 1, 2, [0.0], 0.5)
    with mock.patch("urllib.request.urlopen") as urlopen:
        er.notify_callback("http://api/cb", results)
    req = urlopen.call_args[0][0]
    body = json.loads(req.data)
    assert set(body) == {"eval_id", "status", "scenarios_total",
                         "scenarios_passed", "pass_rate", "results"}
    assert body["pass_rate"] == 0.5
