"""
Evaluate a trained reach policy across varying difficulty levels.

Tests reaching at different distances and heights to measure
generalization and robustness.

Usage:
    python evaluate_reach.py \
        --model-path training_output/policy.zip \
        --episodes 100
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from fleetos_reach_env import register_reach_env, FleetOSReachEnv

register_reach_env()

SCENARIOS = {
    "close_range": {"distance": (0.1, 0.25), "height": (0.9, 1.3)},
    "mid_range": {"distance": (0.25, 0.4), "height": (0.8, 1.3)},
    "far_range": {"distance": (0.35, 0.5), "height": (0.7, 1.4)},
    "low_target": {"distance": (0.15, 0.4), "height": (0.5, 0.8)},
    "high_target": {"distance": (0.15, 0.4), "height": (1.2, 1.5)},
}


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("FleetOS-Reach-v1")
    model = SAC.load(args.model_path, env=env)

    all_results = {}
    total_success = 0
    total_episodes = 0

    for scenario_name, bounds in SCENARIOS.items():
        successes = 0
        distances = []

        for _ in range(args.episodes):
            obs, _ = env.reset()

            # Override goal to match scenario bounds
            robot_x = env.unwrapped.data.qpos[env.unwrapped._root_qpos_adr]
            robot_y = env.unwrapped.data.qpos[env.unwrapped._root_qpos_adr + 1]
            dist = env.unwrapped.np_random.uniform(*bounds["distance"])
            angle = env.unwrapped.np_random.uniform(-np.pi / 2, 0)  # right side
            h = env.unwrapped.np_random.uniform(*bounds["height"])
            goal = np.array([
                robot_x + dist * np.cos(angle),
                robot_y + dist * np.sin(angle),
                h,
            ])
            env.unwrapped._goal = goal
            obs["desired_goal"] = goal.copy()

            done = False
            final_dist = float("inf")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_dist = float(np.linalg.norm(
                    obs["achieved_goal"] - obs["desired_goal"]
                ))

            if info.get("is_success"):
                successes += 1
            distances.append(final_dist)

        rate = successes / args.episodes
        all_results[scenario_name] = {
            "success_rate": round(rate, 3),
            "mean_final_distance": round(float(np.mean(distances)), 4),
            "median_final_distance": round(float(np.median(distances)), 4),
            "episodes": args.episodes,
        }
        total_success += successes
        total_episodes += args.episodes
        print(f"[eval] {scenario_name}: {rate:.1%} success, "
              f"mean_dist={np.mean(distances):.3f}m")

    env.close()

    overall_rate = total_success / total_episodes
    results = {
        "model_path": args.model_path,
        "overall_success_rate": round(overall_rate, 3),
        "total_episodes": total_episodes,
        "scenarios": all_results,
        "pass": overall_rate >= args.pass_threshold,
        "pass_threshold": args.pass_threshold,
    }

    results_path = output_dir / "reach_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[eval] Overall: {overall_rate:.1%} success "
          f"({'PASS' if results['pass'] else 'FAIL'}, "
          f"threshold={args.pass_threshold:.0%})")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate reach policy")
    p.add_argument("--model-path", required=True)
    p.add_argument("--episodes", type=int, default=50, help="Episodes per scenario")
    p.add_argument("--pass-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", default="eval_output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
