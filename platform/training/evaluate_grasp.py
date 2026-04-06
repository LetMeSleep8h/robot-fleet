"""
Evaluate a trained grasp policy across varying difficulty levels.

Tests grasping different objects at varying positions to measure
generalization and robustness.

Usage:
    python evaluate_grasp.py \
        --model-path training_output/policy.zip \
        --episodes 50
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from fleetos_grasp_env import register_grasp_env, FleetOSGraspEnv, OBJECT_NAMES

register_grasp_env()

SCENARIOS = {
    "red_box_standard": {
        "target_object": "red_box",
        "lift_range": (0.10, 0.15),
    },
    "blue_cylinder_standard": {
        "target_object": "blue_cylinder",
        "lift_range": (0.10, 0.15),
    },
    "red_box_high_lift": {
        "target_object": "red_box",
        "lift_range": (0.15, 0.25),
    },
    "blue_cylinder_high_lift": {
        "target_object": "blue_cylinder",
        "lift_range": (0.15, 0.25),
    },
}


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("FleetOS-Grasp-v1")
    model = SAC.load(args.model_path, env=env)

    all_results = {}
    total_success = 0
    total_episodes = 0

    for scenario_name, config in SCENARIOS.items():
        successes = 0
        distances = []
        grasp_steps = []

        for _ in range(args.episodes):
            obs, _ = env.reset()

            # Override target object and goal
            env.unwrapped._target_object = config["target_object"]
            obj_pos = env.unwrapped._get_object_pos()
            lift = env.unwrapped.np_random.uniform(*config["lift_range"])
            goal = obj_pos.copy()
            goal[2] += lift
            env.unwrapped._goal = goal
            env.unwrapped._object_initial_pos = obj_pos.copy()
            obs["desired_goal"] = goal.copy()

            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            final_dist = float(info.get("object_distance", float("inf")))
            distances.append(final_dist)

            if info.get("is_success"):
                successes += 1
                grasp_steps.append(steps)

        rate = successes / args.episodes
        all_results[scenario_name] = {
            "success_rate": round(rate, 3),
            "mean_final_distance": round(float(np.mean(distances)), 4),
            "median_final_distance": round(float(np.median(distances)), 4),
            "mean_grasp_steps": round(float(np.mean(grasp_steps)), 1) if grasp_steps else 0,
            "episodes": args.episodes,
            "target_object": config["target_object"],
        }
        total_success += successes
        total_episodes += args.episodes
        print(
            f"[eval] {scenario_name}: {rate:.1%} success, "
            f"mean_dist={np.mean(distances):.3f}m"
        )

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

    results_path = output_dir / "grasp_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n[eval] Overall: {overall_rate:.1%} success "
        f"({'PASS' if results['pass'] else 'FAIL'}, "
        f"threshold={args.pass_threshold:.0%})"
    )

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate grasp policy")
    p.add_argument("--model-path", required=True)
    p.add_argument("--episodes", type=int, default=50, help="Episodes per scenario")
    p.add_argument("--pass-threshold", type=float, default=0.3)
    p.add_argument("--output-dir", default="eval_output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
