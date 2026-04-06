"""
SB3 SAC+HER training script for FleetOS humanoid grasping.

Trains a goal-conditioned policy that grasps objects on the workbench
and lifts them to target positions. Uses Hindsight Experience Replay
for sample-efficient learning from sparse rewards.

Optionally pre-fills the replay buffer with scripted demonstrations
or loads BC-pretrained weights as a base model.

Usage:
    # Train from scratch
    python train_grasp.py \
        --job-id GRASP_001 \
        --timesteps 1000000 \
        --s3-endpoint minio:9000

    # With scripted demonstrations in replay buffer
    python train_grasp.py \
        --job-id GRASP_002 \
        --timesteps 1000000 \
        --generate-demos 1000 \
        --s3-endpoint minio:9000

    # Fine-tune from BC-pretrained model
    python train_grasp.py \
        --job-id GRASP_003 \
        --timesteps 500000 \
        --base-model training/BC_GRASP_001/policy.zip \
        --s3-endpoint minio:9000
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her import HerReplayBuffer

from fleetos_grasp_env import register_grasp_env

register_grasp_env()

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "")


def push_metrics(job_id: str, metrics: dict):
    """Push training metrics to Prometheus Pushgateway (if configured)."""
    if not PUSHGATEWAY_URL:
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        for key, value in metrics.items():
            g = Gauge(
                f"fleetos_training_{key}", key, ["job_id"], registry=registry
            )
            g.labels(job_id=job_id).set(value)
        push_to_gateway(
            PUSHGATEWAY_URL, job=f"training-{job_id}", registry=registry
        )
    except Exception as e:
        print(f"[pushgateway] Failed: {e}")


class GraspMetricsCallback(BaseCallback):
    """Logs grasp-specific metrics: success rate, object distance, gripper state."""

    def __init__(self, metrics_path: str, job_id: str = "", verbose=0):
        super().__init__(verbose)
        self.metrics_path = metrics_path
        self.job_id = job_id
        self.successes = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "is_success" in info:
                    self.successes.append(float(info["is_success"]))
                    self.episode_count += 1

                    if self.episode_count % 50 == 0:
                        self._flush()
        return True

    def _flush(self):
        recent = self.successes[-100:] if self.successes else []
        summary = {
            "total_episodes": self.episode_count,
            "success_rate": float(np.mean(recent)) if recent else 0.0,
            "total_successes": int(sum(self.successes)),
            "total_timesteps": self.num_timesteps,
        }
        with open(self.metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        push_metrics(self.job_id, {
            "success_rate": summary["success_rate"],
            "timesteps": self.num_timesteps,
            "episodes": self.episode_count,
        })

        if self.verbose:
            print(
                f"[grasp] ep={self.episode_count} "
                f"success_rate={summary['success_rate']:.1%} "
                f"steps={self.num_timesteps}"
            )

    def _on_training_end(self):
        self._flush()


def generate_scripted_demos(env, n_demos: int) -> list[dict]:
    """Generate demonstrations using a simple scripted controller.

    Strategy: move hand toward object -> close gripper -> lift.
    Produces noisy but mostly successful grasp demonstrations.
    """
    demos = []
    raw_env = env.envs[0] if hasattr(env, 'envs') else env

    for ep in range(n_demos):
        obs, _ = raw_env.reset()
        done = False
        episode_transitions = []

        # Get target object position from observation
        # Object positions are at indices 66-71 in the 86-dim observation
        obj_pos = obs["achieved_goal"]  # current object position
        goal = obs["desired_goal"]      # lift target

        step = 0
        phase = "approach"  # approach -> close -> lift

        while not done:
            hand_pos = raw_env.data.site_xpos[raw_env._hand_site_id].copy()
            action = np.zeros(raw_env.action_space.shape[0])

            if phase == "approach":
                # Move hand toward object
                diff = obj_pos - hand_pos
                # Map world-space direction to rough joint commands
                # Right shoulder1 controls forward/back, shoulder2 up/down
                action[16] = np.clip(diff[0] * 2.0, -0.5, 0.5)   # right_shoulder1
                action[17] = np.clip(-diff[2] * 2.0, -0.5, 0.5)  # right_shoulder2
                action[18] = np.clip(diff[1] * 2.0, -0.5, 0.5)   # right_elbow
                # Open gripper
                action[-4] = 0.0  # right_gripper_l open
                action[-3] = 0.0  # right_gripper_r open

                if np.linalg.norm(diff) < 0.08:
                    phase = "close"

            elif phase == "close":
                # Close gripper around object
                action[-4] = 0.7  # right_gripper_l close
                action[-3] = 0.7  # right_gripper_r close
                step += 1
                if step > 20:
                    phase = "lift"
                    step = 0

            elif phase == "lift":
                # Lift upward
                action[16] = 0.0
                action[17] = -0.5  # shoulder2 up
                action[-4] = 0.7   # keep gripper closed
                action[-3] = 0.7

            # Add noise for diversity
            action += raw_env.np_random.normal(0, 0.05, size=action.shape)
            action = np.clip(action, raw_env.action_space.low, raw_env.action_space.high)

            next_obs, reward, terminated, truncated, info = raw_env.step(action)
            done = terminated or truncated

            episode_transitions.append({
                "obs": obs["observation"].tolist(),
                "action": action.tolist(),
                "reward": float(reward),
                "next_obs": next_obs["observation"].tolist(),
                "done": done,
            })

            obs = next_obs
            obj_pos = obs["achieved_goal"]

        demos.extend(episode_transitions)

        if (ep + 1) % 100 == 0:
            print(f"[demos] Generated {ep + 1}/{n_demos} episodes "
                  f"({len(demos)} transitions)")

    print(f"[demos] Total: {len(demos)} transitions from {n_demos} episodes")
    return demos


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_id = "FleetOS-Grasp-v1"
    print(f"[grasp] Starting SAC+HER training on {env_id}")
    print(f"[grasp] Job ID: {args.job_id}")
    print(f"[grasp] Timesteps: {args.timesteps}")

    env = DummyVecEnv([lambda: gym.make(env_id)])

    if args.base_model:
        base_path = _resolve_model_path(args, args.base_model)
        if base_path:
            print(f"[grasp] Fine-tuning from: {base_path}")
            model = SAC.load(base_path, env=env, device=args.device)
        else:
            print(f"[grasp] Base model not found, training from scratch")
            model = _create_model(env, args)
    else:
        model = _create_model(env, args)

    # Pre-fill replay buffer with scripted demonstrations
    if args.generate_demos > 0:
        print(f"[grasp] Generating {args.generate_demos} scripted demonstrations...")
        demo_env = gym.make(env_id)
        demos = generate_scripted_demos(demo_env, args.generate_demos)
        demo_env.close()

        obs_dim = 86
        act_dim = env.action_space.shape[0]
        added = 0
        for t in demos:
            try:
                obs_arr = np.array(t["obs"], dtype=np.float32)
                act_arr = np.array(t["action"], dtype=np.float32)
                next_obs_arr = np.array(t["next_obs"], dtype=np.float32)
                if obs_arr.shape[0] == obs_dim and act_arr.shape[0] == act_dim:
                    model.replay_buffer.add(
                        obs={"observation": obs_arr,
                             "achieved_goal": np.zeros(3, dtype=np.float32),
                             "desired_goal": np.zeros(3, dtype=np.float32)},
                        next_obs={"observation": next_obs_arr,
                                  "achieved_goal": np.zeros(3, dtype=np.float32),
                                  "desired_goal": np.zeros(3, dtype=np.float32)},
                        action=act_arr,
                        reward=np.array([t["reward"]]),
                        done=np.array([t["done"]]),
                        infos=[{"is_success": False}],
                    )
                    added += 1
            except Exception:
                continue
        print(f"[grasp] Added {added} demo transitions to replay buffer")

    metrics_path = str(output_dir / "metrics.json")
    callback = GraspMetricsCallback(
        metrics_path, job_id=args.job_id, verbose=1
    )

    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    train_duration = time.time() - start_time

    # Save
    model_path = str(output_dir / "policy")
    model.save(model_path)

    # Evaluate
    success_count = 0
    eval_episodes = 50
    eval_env = gym.make(env_id)
    grasp_times = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            steps += 1
        if info.get("is_success"):
            success_count += 1
            grasp_times.append(steps)
    eval_env.close()

    success_rate = success_count / eval_episodes

    results = {
        "job_id": args.job_id,
        "status": "completed",
        "task_type": "grasp",
        "algorithm": "SAC+HER",
        "model_path": model_path + ".zip",
        "artifact_url": f"training/{args.job_id}/policy.zip",
        "environment": env_id,
        "train_duration_seconds": round(train_duration, 1),
        "total_timesteps": args.timesteps,
        "eval_success_rate": round(success_rate, 3),
        "eval_episodes": eval_episodes,
        "mean_grasp_steps": round(float(np.mean(grasp_times)), 1) if grasp_times else 0,
        "demos_used": args.generate_demos,
        "hyperparameters": {
            "algorithm": "SAC",
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "her_n_sampled_goal": 4,
            "her_strategy": "future",
            "net_arch": [256, 256],
            "distance_threshold": 0.05,
        },
    }
    with open(str(output_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[grasp] Training complete: success_rate={success_rate:.1%}, "
        f"duration={train_duration:.0f}s"
    )

    if args.s3_endpoint:
        _upload_to_s3(args, output_dir)

    if args.callback_url:
        _notify_callback(args.callback_url, results)

    return results


def _create_model(env, args) -> SAC:
    """Create SAC model with HER replay buffer for grasping."""
    return SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=0.95,
        tau=0.05,
        learning_starts=1000,
        verbose=1,
        device=args.device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )


def _resolve_model_path(args, artifact_url: str) -> str | None:
    if not args.s3_endpoint:
        return None
    try:
        from minio import Minio

        client = Minio(
            args.s3_endpoint, access_key=args.s3_access_key,
            secret_key=args.s3_secret_key, secure=False,
        )
        local_path = Path(args.output_dir) / "base_policy.zip"
        key = artifact_url.replace(f"s3://{args.s3_bucket}/", "")
        client.fget_object(args.s3_bucket, key, str(local_path))
        return str(local_path).replace(".zip", "")
    except Exception as e:
        print(f"[s3] Failed to download base model: {e}")
        return None


def _upload_to_s3(args, output_dir: Path):
    try:
        from minio import Minio

        client = Minio(
            args.s3_endpoint, access_key=args.s3_access_key,
            secret_key=args.s3_secret_key, secure=False,
        )
        if not client.bucket_exists(args.s3_bucket):
            client.make_bucket(args.s3_bucket)
        for f in output_dir.iterdir():
            if f.is_file():
                key = f"training/{args.job_id}/{f.name}"
                client.fput_object(args.s3_bucket, key, str(f))
                print(f"[s3] Uploaded {key}")
    except Exception as e:
        print(f"[s3] Upload failed: {e}")


def _notify_callback(url: str, results: dict):
    try:
        import urllib.request

        req = urllib.request.Request(
            url, data=json.dumps(results).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[callback] Notified {url}")
    except Exception as e:
        print(f"[callback] Failed: {e}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train grasping policy with SAC+HER"
    )
    p.add_argument("--job-id", default=f"grasp-{int(time.time())}")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--base-model", default=None)
    p.add_argument("--generate-demos", type=int, default=0,
                   help="Number of scripted demonstration episodes to generate")
    p.add_argument("--output-dir", default="training_output")
    p.add_argument("--device", default="auto")
    p.add_argument("--s3-endpoint", default=os.environ.get("S3_ENDPOINT", ""))
    p.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", "fleetos-models"))
    p.add_argument("--s3-access-key", default=os.environ.get("S3_ACCESS_KEY", "fleetos"))
    p.add_argument("--s3-secret-key", default=os.environ.get("S3_SECRET_KEY", "fleetos123"))
    p.add_argument("--callback-url", default=os.environ.get("TRAINING_CALLBACK_URL", ""))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
