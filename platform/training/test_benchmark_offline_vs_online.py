"""Smoke test for the online-vs-offline training benchmark."""

import json

import benchmark_offline_vs_online as bench


def test_tiny_benchmark_reports_both_paths(capsys):
    assert bench.main(["--timesteps", "64"]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    for key in ("online_seconds", "offline_seconds", "online_sim_steps_per_s",
                "offline_updates_per_s", "cpu_count", "online_gradient_updates", "offline_gradient_steps"):
        assert key in result, key
    assert result["online_seconds"] > 0 and result["offline_seconds"] > 0
    # Both paths do the same number of gradient updates, so time differences
    # come from simulation, not from doing less learning.
    assert result["offline_gradient_steps"] == result["online_gradient_updates"]
