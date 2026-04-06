"""
SB3-contrib offline RL training for FleetOS humanoid.

Trains a policy purely from collected experience data in S3 using
Conservative Q-Learning (CQL) via TQC from sb3-contrib, without
requiring any environment interaction during training.

Experience files are NDJSON in S3:
  {obs: [86], action: [25], reward: float, next_obs: [86], done: bool}

Usage:
    # Train from collected experience
    python train_offline.py \
        --job-id OFFLINE_001 \
        --experience-prefix experience/ \
        --gradient-steps 100000 \
        --s3-endpoint minio:9000 \
        --s3-bucket fleetos-models

    # With custom CQL conservatism
    python train_offline.py \
        --job-id OFFLINE_002 \
        --experience-prefix experience/ \
        --gradient-steps 200000 \
        --cql-alpha 2.0 \
        --s3-endpoint minio:9000
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from fleetos_env import register_fleetos_env

register_fleetos_env()


class CQLTrainer:
    """Conservative Q-Learning wrapper around SB3 SAC.

    Adds a CQL-style conservative penalty to SAC's Q-function updates,
    penalizing Q-values on out-of-distribution actions to prevent
    overestimation from offline data.
    """

    def __init__(self, model: SAC, cql_alpha: float = 1.0):
        self.model = model
        self.cql_alpha = cql_alpha

    def train_offline(self, gradient_steps: int, batch_size: int = 256,
                      log_interval: int = 1000):
        """Run offline gradient updates with CQL penalty.

        Instead of collecting new rollouts, we repeatedly sample from
        the pre-filled replay buffer and update the SAC critic with
        an additional conservative penalty term.
        """
        model = self.model
        logger_losses = []

        for step in range(1, gradient_steps + 1):
            # Sample batch from pre-filled replay buffer
            replay_data = model.replay_buffer.sample(batch_size)

            # Standard SAC critic update
            with torch.no_grad():
                next_actions, next_log_prob = model.actor.action_log_prob(
                    replay_data.next_observations
                )
                targets_q1, targets_q2 = model.critic_target(
                    replay_data.next_observations, next_actions
                )
                target_q = torch.min(targets_q1, targets_q2)
                target_q = target_q - model.ent_coef * next_log_prob
                target_q = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * model.gamma * target_q
                )

            current_q1, current_q2 = model.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )

            # CQL conservative penalty: penalize high Q-values on random actions
            # to prevent overestimation of unseen state-action pairs
            obs = replay_data.observations
            n_random = 10
            random_actions = torch.FloatTensor(
                obs.shape[0] * n_random, model.action_space.shape[0]
            ).uniform_(
                model.action_space.low[0], model.action_space.high[0]
            ).to(obs.device)

            obs_repeat = obs.unsqueeze(1).repeat(1, n_random, 1).view(
                -1, obs.shape[-1]
            )

            random_q1, random_q2 = model.critic(obs_repeat, random_actions)
            random_q1 = random_q1.view(obs.shape[0], n_random)
            random_q2 = random_q2.view(obs.shape[0], n_random)

            cql_penalty = (
                torch.logsumexp(random_q1, dim=1).mean()
                + torch.logsumexp(random_q2, dim=1).mean()
                - current_q1.mean()
                - current_q2.mean()
            )

            total_loss = critic_loss + self.cql_alpha * cql_penalty

            model.critic.optimizer.zero_grad()
            total_loss.backward()
            model.critic.optimizer.step()

            # Actor update (standard SAC)
            actions_pi, log_prob = model.actor.action_log_prob(obs)
            q1_pi, q2_pi = model.critic(obs, actions_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (model.ent_coef * log_prob - min_q_pi).mean()

            model.actor.optimizer.zero_grad()
            actor_loss.backward()
            model.actor.optimizer.step()

            # Soft update target networks
            for param, target_param in zip(
                model.critic.parameters(), model.critic_target.parameters()
            ):
                target_param.data.copy_(
                    0.005 * param.data + 0.995 * target_param.data
                )

            logger_losses.append(float(total_loss.item()))

            if step % log_interval == 0:
                avg_loss = np.mean(logger_losses[-log_interval:])
                print(
                    f"[cql] step={step}/{gradient_steps} "
                    f"loss={avg_loss:.4f} "
                    f"cql_penalty={float(cql_penalty):.4f}"
                )

        return {"mean_loss": float(np.mean(logger_losses))}


def load_experience_from_s3(args) -> list[dict]:
    """Download and parse NDJSON experience files from S3."""
    from minio import Minio

    client = Minio(
        args.s3_endpoint,
        access_key=args.s3_access_key,
        secret_key=args.s3_secret_key,
        secure=False,
    )

    prefix = args.experience_prefix
    objects = list(
        client.list_objects(args.s3_bucket, prefix=prefix, recursive=True)
    )
    print(f"[experience] Found {len(objects)} files under {prefix}")

    transitions = []
    for obj in objects[:100]:  # limit to avoid OOM
        try:
            response = client.get_object(args.s3_bucket, obj.object_name)
            for line in response.read().decode().strip().split("\n"):
                if line:
                    transitions.append(json.loads(line))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"[experience] Skip {obj.object_name}: {e}")

    print(f"[experience] Loaded {len(transitions)} transitions")
    return transitions


def fill_replay_buffer(buffer: ReplayBuffer, transitions: list[dict],
                       obs_dim: int, act_dim: int):
    """Populate SB3 ReplayBuffer from NDJSON transitions."""
    added = 0
    for t in transitions:
        obs = np.array(t.get("obs", [0.0] * obs_dim), dtype=np.float32)
        action = np.array(t.get("action", [0.0] * act_dim), dtype=np.float32)
        reward = float(t.get("reward", 0.0))
        next_obs = np.array(
            t.get("next_obs", [0.0] * obs_dim), dtype=np.float32
        )
        done = bool(t.get("done", False))

        if obs.shape[0] != obs_dim or action.shape[0] != act_dim:
            continue

        buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=np.array([reward]),
            done=np.array([done]),
            infos=[{}],
        )
        added += 1

    print(f"[buffer] Added {added}/{len(transitions)} transitions to replay buffer")
    return added


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_id = args.env_id
    print(f"[offline] Starting CQL offline training")
    print(f"[offline] Job ID: {args.job_id}")
    print(f"[offline] Gradient steps: {args.gradient_steps}")
    print(f"[offline] CQL alpha: {args.cql_alpha}")
    print(f"[offline] Experience: {args.experience_prefix}")

    # Load experience data
    transitions = load_experience_from_s3(args)
    if len(transitions) < 100:
        print("[offline] ERROR: Not enough experience data (<100 transitions)")
        results = {
            "job_id": args.job_id,
            "status": "failed",
            "error_message": f"Insufficient experience data: {len(transitions)} transitions",
        }
        with open(str(output_dir / "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Create environment (for action/obs spaces and evaluation only)
    env = DummyVecEnv([lambda: gym.make(env_id)])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create SAC model with empty replay buffer
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=max(len(transitions) * 2, 100_000),
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.005,
        verbose=0,
        device=args.device,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # Fill replay buffer with offline data
    fill_replay_buffer(model.replay_buffer, transitions, obs_dim, act_dim)

    # Train with CQL
    start_time = time.time()
    cql_trainer = CQLTrainer(model, cql_alpha=args.cql_alpha)
    train_stats = cql_trainer.train_offline(
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )
    train_duration = time.time() - start_time

    # Save model (compatible with SB3 SAC.load())
    model_path = str(output_dir / "policy")
    model.save(model_path)

    # Evaluate in environment
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)

    results = {
        "job_id": args.job_id,
        "status": "completed",
        "model_path": model_path + ".zip",
        "artifact_url": f"training/{args.job_id}/policy.zip",
        "environment": env_id,
        "train_duration_seconds": round(train_duration, 1),
        "gradient_steps": args.gradient_steps,
        "experience_transitions": len(transitions),
        "eval_mean_reward": round(float(mean_reward), 2),
        "eval_std_reward": round(float(std_reward), 2),
        "mean_training_loss": round(train_stats["mean_loss"], 4),
        "hyperparameters": {
            "algorithm": "CQL",
            "cql_alpha": args.cql_alpha,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gamma": 0.99,
            "net_arch": [256, 256],
        },
    }
    with open(str(output_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[offline] Training complete: reward={mean_reward:.1f}+/-{std_reward:.1f}, "
        f"duration={train_duration:.0f}s, loss={train_stats['mean_loss']:.4f}"
    )

    env.close()

    if args.s3_endpoint:
        upload_to_s3(args, output_dir)

    if args.callback_url:
        notify_callback(args.callback_url, results)

    return results


def upload_to_s3(args, output_dir: Path):
    """Upload training artifacts to S3/MinIO."""
    try:
        from minio import Minio

        client = Minio(
            args.s3_endpoint,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
            secure=False,
        )
        bucket = args.s3_bucket
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        prefix = f"training/{args.job_id}"
        for fpath in output_dir.iterdir():
            if fpath.is_file():
                key = f"{prefix}/{fpath.name}"
                client.fput_object(bucket, key, str(fpath))
                print(f"[s3] Uploaded {key}")
    except Exception as e:
        print(f"[s3] Upload failed (non-fatal): {e}")


def notify_callback(url: str, results: dict):
    """POST training results to the platform callback endpoint."""
    try:
        import urllib.request

        data = json.dumps(results).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[callback] Notified {url}")
    except Exception as e:
        print(f"[callback] Failed (non-fatal): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CQL offline RL training")
    parser.add_argument("--job-id", required=True, help="Training job ID")
    parser.add_argument("--experience-prefix", required=True,
                        help="S3 prefix for experience NDJSON files")
    parser.add_argument("--gradient-steps", type=int, default=100_000,
                        help="Number of gradient update steps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cql-alpha", type=float, default=1.0,
                        help="CQL conservatism coefficient (higher = more conservative)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env-id", default="FleetOS-Humanoid-v2",
                        help="Environment for evaluation (not used during training)")
    parser.add_argument("--output-dir", default="/tmp/training-output")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", ""))
    parser.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", "fleetos-models"))
    parser.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "fleetos"))
    parser.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "fleetos123"))
    parser.add_argument("--callback-url", default=os.getenv("TRAINING_CALLBACK_URL", ""))
    args = parser.parse_args()

    train(args)
