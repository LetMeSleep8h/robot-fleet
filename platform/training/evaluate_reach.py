"""
Evaluate a trained reach policy across varying difficulty levels.

Tests reaching at different distances and heights to measure
generalization and robustness.

Results follow the same contract as evaluate_policy.py so the platform can
gate any task the same way:
  - written to <output-dir>/eval_results.json with top-level `pass_rate` (0–1)
    and `mean_reward`
  - uploaded to s3://<bucket>/evaluations/<eval-id>/eval_results.json
  - POSTed to EVAL_CALLBACK_URL (eval_id, status, scenarios_total,
    scenarios_passed, pass_rate, results)

Usage:
    python evaluate_reach.py \
        --eval-id reach-001 \
        --model-path s3://fleetos-models/training/reach-001/policy.zip \
        --episodes 20
"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from fleetos_reach_env import register_reach_env

register_reach_env()

SCENARIOS = {
    "close_range": {"distance": (0.1, 0.25), "height": (0.9, 1.3)},
    "mid_range": {"distance": (0.25, 0.4), "height": (0.8, 1.3)},
    "far_range": {"distance": (0.35, 0.5), "height": (0.7, 1.4)},
    "low_target": {"distance": (0.15, 0.4), "height": (0.5, 0.8)},
    "high_target": {"distance": (0.15, 0.4), "height": (1.2, 1.5)},
}

RESULTS_FILENAME = "eval_results.json"


def s3_key_from_model_path(model_path: str, bucket: str) -> str | None:
    """Return the S3 key for an s3:// URL or bare training/ key, else None (local path)."""
    if model_path.startswith(f"s3://{bucket}/"):
        return model_path[len(f"s3://{bucket}/"):]
    if model_path.startswith("training/"):
        return model_path
    return None


def _s3_client(args):
    from minio import Minio
    return Minio(args.s3_endpoint, access_key=args.s3_access_key,
                 secret_key=args.s3_secret_key, secure=False)


def resolve_model(args) -> str:
    """Download the model from S3 when model-path is an S3 location."""
    key = s3_key_from_model_path(args.model_path, args.s3_bucket)
    if key is None:
        return args.model_path
    local_dir = Path(args.output_dir) / "model"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = str(local_dir / "policy.zip")
    _s3_client(args).fget_object(args.s3_bucket, key, local_path)
    print(f"[eval] Downloaded model from {key}")
    return local_path


def build_results(eval_id: str, model_path: str, scenario_stats: dict,
                  passed: int, total: int, rewards: list[float],
                  pass_threshold: float) -> dict:
    """Assemble results in the shared eval contract (plus reach-specific extras)."""
    pass_rate = passed / total if total > 0 else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    return {
        "eval_id": eval_id,
        "status": "completed",
        "model_path": model_path,
        "scenarios_total": total,
        "scenarios_passed": passed,
        "pass_rate": round(pass_rate, 4),
        "mean_reward": round(mean_reward, 2),
        "results": scenario_stats,
        # Reach-specific fields kept for compatibility with earlier reports.
        "overall_success_rate": round(pass_rate, 3),
        "total_episodes": total,
        "pass": pass_rate >= pass_threshold,
        "pass_threshold": pass_threshold,
    }


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("FleetOS-Reach-v1")
    model = SAC.load(resolve_model(args), env=env)

    scenario_stats = {}
    total_success = 0
    total_episodes = 0
    episode_rewards: list[float] = []

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
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                done = terminated or truncated
                final_dist = float(np.linalg.norm(
                    obs["achieved_goal"] - obs["desired_goal"]
                ))

            if info.get("is_success"):
                successes += 1
            distances.append(final_dist)
            episode_rewards.append(ep_reward)

        rate = successes / args.episodes
        scenario_stats[scenario_name] = {
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

    results = build_results(args.eval_id, args.model_path, scenario_stats,
                            total_success, total_episodes, episode_rewards,
                            args.pass_threshold)

    with open(output_dir / RESULTS_FILENAME, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[eval] Overall: {results['pass_rate']:.1%} success "
          f"({'PASS' if results['pass'] else 'FAIL'}, "
          f"threshold={args.pass_threshold:.0%})")

    if args.s3_endpoint:
        upload_results(args, output_dir)
    if args.callback_url:
        notify_callback(args.callback_url, results)

    return results


def upload_results(args, output_dir: Path):
    """Upload eval files to evaluations/<eval-id>/. Fatal on failure — the
    platform's evaluation gate reads eval_results.json from S3."""
    client = _s3_client(args)
    for fpath in output_dir.iterdir():
        if not fpath.is_file():
            continue  # skip the downloaded model/ directory
        key = f"evaluations/{args.eval_id}/{fpath.name}"
        client.fput_object(args.s3_bucket, key, str(fpath))
        print(f"[s3] Uploaded {key}")


def notify_callback(url: str, results: dict):
    payload = {k: results[k] for k in (
        "eval_id", "status", "scenarios_total", "scenarios_passed", "pass_rate", "results",
    )}
    try:
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[callback] Notified {url}")
    except Exception as e:
        print(f"[callback] Failed (non-fatal): {e}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate reach policy")
    p.add_argument("--eval-id", default=os.getenv("EVAL_ID", "reach-eval"))
    p.add_argument("--model-path", required=True,
                   help="Local path, s3://<bucket>/<key>, or bare training/<job>/policy.zip key")
    p.add_argument("--episodes", type=int, default=50, help="Episodes per scenario")
    p.add_argument("--scenarios", type=int, default=0,
                   help="Total episodes across all scenarios (overrides --episodes when > 0)")
    p.add_argument("--pass-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", default="eval_output")
    p.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", ""))
    p.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", "fleetos-models"))
    p.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "fleetos"))
    p.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "fleetos123"))
    p.add_argument("--callback-url", default=os.getenv("EVAL_CALLBACK_URL", ""))
    args = p.parse_args(argv)
    if args.scenarios > 0:
        args.episodes = max(1, args.scenarios // len(SCENARIOS))
    return args


if __name__ == "__main__":
    evaluate(parse_args())
