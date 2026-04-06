"""
Behavior Cloning (supervised pre-training) for FleetOS humanoid.

Trains a policy via supervised learning on (observation, action) pairs
from expert demonstrations. The output is an SB3-compatible checkpoint
that can be used as --base-model for RL fine-tuning.

Two data sources:
  1. experience: Filter high-reward transitions from S3 NDJSON data
  2. expert: Roll out a trained policy to generate demonstrations

Usage:
    # From collected experience (filter top-performing transitions)
    python train_bc.py \
        --job-id BC_001 \
        --task-type locomotion \
        --demo-source experience \
        --experience-prefix experience/ \
        --min-reward 2.0 \
        --s3-endpoint minio:9000

    # From expert policy rollouts
    python train_bc.py \
        --job-id BC_002 \
        --task-type locomotion \
        --demo-source expert \
        --expert-model training/PPO_001/policy.zip \
        --demo-episodes 200 \
        --s3-endpoint minio:9000

    # Then fine-tune with RL:
    python train_locomotion.py --base-model training/BC_001/policy.zip ...
"""

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fleetos_env import register_fleetos_env
from fleetos_reach_env import register_reach_env

register_fleetos_env()
register_reach_env()

# Env IDs and algorithms per task type
TASK_CONFIG = {
    "locomotion": {"env_id": "FleetOS-Humanoid-v2", "algorithm": "PPO"},
    "reach": {"env_id": "FleetOS-Reach-v1", "algorithm": "SAC"},
    "grasp": {"env_id": "FleetOS-Grasp-v1", "algorithm": "SAC"},
}


class BCPolicy(nn.Module):
    """MLP policy matching SB3's architecture for weight transfer."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int] = None):
        super().__init__()
        hidden = hidden or [256, 256]
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def load_demos_from_experience(args) -> tuple[np.ndarray, np.ndarray]:
    """Load demonstrations from S3 NDJSON, filtering by reward threshold."""
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
    print(f"[bc] Found {len(objects)} experience files under {prefix}")

    observations = []
    actions = []

    for obj in objects[:100]:
        try:
            response = client.get_object(args.s3_bucket, obj.object_name)
            for line in response.read().decode().strip().split("\n"):
                if not line:
                    continue
                t = json.loads(line)
                reward = float(t.get("reward", 0.0))
                if reward >= args.min_reward:
                    observations.append(t["obs"])
                    actions.append(t["action"])
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"[bc] Skip {obj.object_name}: {e}")

    print(f"[bc] Loaded {len(observations)} demonstrations "
          f"(reward >= {args.min_reward})")

    return (
        np.array(observations, dtype=np.float32),
        np.array(actions, dtype=np.float32),
    )


def load_demos_from_expert(args, env) -> tuple[np.ndarray, np.ndarray]:
    """Generate demonstrations by rolling out an expert policy."""
    # Download expert model from S3
    expert_path = _download_model(args, args.expert_model)
    if expert_path is None:
        raise ValueError(f"Could not download expert model: {args.expert_model}")

    task_cfg = TASK_CONFIG[args.task_type]
    if task_cfg["algorithm"] == "PPO":
        expert = PPO.load(expert_path, env=env)
    else:
        expert = SAC.load(expert_path, env=env)

    observations = []
    actions = []
    raw_env = gym.make(task_cfg["env_id"])

    for ep in range(args.demo_episodes):
        obs, _ = raw_env.reset()
        done = False
        while not done:
            # For Dict obs (reach/grasp), extract the observation component
            if isinstance(obs, dict):
                flat_obs = obs["observation"]
            else:
                flat_obs = obs

            action, _ = expert.predict(obs, deterministic=True)
            observations.append(flat_obs.copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, _ = raw_env.step(action)
            done = terminated or truncated

    raw_env.close()

    print(f"[bc] Generated {len(observations)} demonstrations "
          f"from {args.demo_episodes} expert episodes")

    return (
        np.array(observations, dtype=np.float32),
        np.array(actions, dtype=np.float32),
    )


def train_bc_policy(obs_data: np.ndarray, act_data: np.ndarray,
                    args) -> BCPolicy:
    """Train BC policy with supervised learning."""
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]

    # Normalize observations
    obs_mean = obs_data.mean(axis=0)
    obs_std = obs_data.std(axis=0) + 1e-8
    obs_normalized = (obs_data - obs_mean) / obs_std

    dataset = TensorDataset(
        torch.FloatTensor(obs_normalized),
        torch.FloatTensor(act_data),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    policy = BCPolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available())
                          else "cpu")
    policy.to(device)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred_actions = policy(obs_batch)
            loss = nn.functional.mse_loss(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"[bc] Epoch {epoch + 1}/{args.epochs} loss={avg_loss:.6f}")

    print(f"[bc] Best training loss: {best_loss:.6f}")
    return policy, obs_mean, obs_std, best_loss


def save_as_sb3_model(bc_policy: BCPolicy, obs_mean: np.ndarray,
                      obs_std: np.ndarray, args, output_dir: Path) -> str:
    """Save BC policy as SB3-compatible checkpoint for fine-tuning.

    Creates an SB3 model (PPO or SAC) and copies the BC-trained weights
    into the policy network so it can be loaded with PPO.load() or SAC.load().
    """
    task_cfg = TASK_CONFIG[args.task_type]
    env_id = task_cfg["env_id"]

    env = DummyVecEnv([lambda: gym.make(env_id)])

    if task_cfg["algorithm"] == "PPO":
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            verbose=0, device=args.device,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )
    else:
        model = SAC(
            "MultiInputPolicy" if args.task_type in ("reach", "grasp") else "MlpPolicy",
            env,
            learning_rate=3e-4, buffer_size=1_000_000,
            verbose=0, device=args.device,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    # Copy BC weights into the SB3 actor network
    bc_state = bc_policy.state_dict()
    actor = model.policy.actor if hasattr(model.policy, 'actor') else model.policy

    # Map BC network layers to SB3 MLP extractor + action net
    # BC net: net.0 (linear), net.1 (relu), net.2 (linear), net.3 (relu), net.4 (output)
    # SB3 PPO: mlp_extractor.policy_net.0, .2 + action_net
    # SB3 SAC: latent_pi.0, .2 + mu (mean action)
    try:
        if task_cfg["algorithm"] == "PPO":
            mlp = model.policy.mlp_extractor.policy_net
            mlp[0].weight.data.copy_(bc_state["net.0.weight"])
            mlp[0].bias.data.copy_(bc_state["net.0.bias"])
            mlp[2].weight.data.copy_(bc_state["net.2.weight"])
            mlp[2].bias.data.copy_(bc_state["net.2.bias"])
            model.policy.action_net.weight.data.copy_(bc_state["net.4.weight"])
            model.policy.action_net.bias.data.copy_(bc_state["net.4.bias"])
        else:
            actor_net = model.policy.actor
            actor_net.latent_pi[0].weight.data.copy_(bc_state["net.0.weight"])
            actor_net.latent_pi[0].bias.data.copy_(bc_state["net.0.bias"])
            actor_net.latent_pi[2].weight.data.copy_(bc_state["net.2.weight"])
            actor_net.latent_pi[2].bias.data.copy_(bc_state["net.2.bias"])
            actor_net.mu.weight.data.copy_(bc_state["net.4.weight"])
            actor_net.mu.bias.data.copy_(bc_state["net.4.bias"])
        print("[bc] Successfully transferred BC weights to SB3 model")
    except Exception as e:
        print(f"[bc] Weight transfer failed (saving BC-only): {e}")
        # Fallback: save raw PyTorch model
        torch.save(bc_policy.state_dict(), str(output_dir / "bc_policy.pt"))

    # Update VecNormalize with BC training stats
    if task_cfg["algorithm"] == "PPO" and hasattr(env, 'obs_rms'):
        env.obs_rms.mean = obs_mean
        env.obs_rms.var = obs_std ** 2
        env.obs_rms.count = len(obs_mean)
        env.save(str(output_dir / "vec_normalize.pkl"))

    model_path = str(output_dir / "policy")
    model.save(model_path)
    env.close()

    return model_path


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = TASK_CONFIG.get(args.task_type)
    if task_cfg is None:
        # Gracefully handle grasp before its env is registered
        print(f"[bc] Unknown task type: {args.task_type}")
        return

    print(f"[bc] Starting behavior cloning for {args.task_type}")
    print(f"[bc] Job ID: {args.job_id}")
    print(f"[bc] Demo source: {args.demo_source}")

    # Create env for expert rollouts and evaluation
    env = DummyVecEnv([lambda: gym.make(task_cfg["env_id"])])

    # Load demonstrations
    start_time = time.time()
    if args.demo_source == "experience":
        obs_data, act_data = load_demos_from_experience(args)
    elif args.demo_source == "expert":
        obs_data, act_data = load_demos_from_expert(args, env)
    else:
        raise ValueError(f"Unknown demo source: {args.demo_source}")

    if len(obs_data) < 100:
        print(f"[bc] ERROR: Not enough demonstrations ({len(obs_data)})")
        results = {
            "job_id": args.job_id,
            "status": "failed",
            "error_message": f"Insufficient demonstrations: {len(obs_data)}",
        }
        with open(str(output_dir / "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Train BC policy
    bc_policy, obs_mean, obs_std, best_loss = train_bc_policy(
        obs_data, act_data, args
    )
    train_duration = time.time() - start_time

    # Save as SB3-compatible model
    model_path = save_as_sb3_model(
        bc_policy, obs_mean, obs_std, args, output_dir
    )

    # Evaluate BC policy in environment
    if task_cfg["algorithm"] == "PPO":
        eval_model = PPO.load(model_path, env=env)
        mean_reward, std_reward = evaluate_policy(
            eval_model, env, n_eval_episodes=20
        )
        eval_metric = {"eval_mean_reward": round(float(mean_reward), 2),
                       "eval_std_reward": round(float(std_reward), 2)}
    else:
        eval_env = gym.make(task_cfg["env_id"])
        eval_model = SAC.load(model_path, env=env)
        successes = 0
        eval_episodes = 20
        for _ in range(eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            if info.get("is_success"):
                successes += 1
        eval_env.close()
        eval_metric = {"eval_success_rate": round(successes / eval_episodes, 3)}

    env.close()

    results = {
        "job_id": args.job_id,
        "status": "completed",
        "task_type": args.task_type,
        "model_path": model_path + ".zip",
        "artifact_url": f"training/{args.job_id}/policy.zip",
        "environment": task_cfg["env_id"],
        "algorithm": "BC",
        "train_duration_seconds": round(train_duration, 1),
        "num_demonstrations": len(obs_data),
        "demo_source": args.demo_source,
        "best_training_loss": round(best_loss, 6),
        **eval_metric,
        "hyperparameters": {
            "algorithm": "BC",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "net_arch": [256, 256],
            "min_reward_filter": args.min_reward,
        },
    }
    with open(str(output_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[bc] Training complete: {eval_metric}, "
          f"duration={train_duration:.0f}s, loss={best_loss:.6f}")

    if args.s3_endpoint:
        _upload_to_s3(args, output_dir)

    if args.callback_url:
        _notify_callback(args.callback_url, results)

    return results


def _download_model(args, artifact_url: str) -> str | None:
    """Download a model from S3."""
    if not args.s3_endpoint:
        return None
    try:
        from minio import Minio

        client = Minio(
            args.s3_endpoint,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
            secure=False,
        )
        local_path = Path(args.output_dir) / "expert_policy.zip"
        key = artifact_url.replace(f"s3://{args.s3_bucket}/", "")
        client.fget_object(args.s3_bucket, key, str(local_path))
        return str(local_path).replace(".zip", "")
    except Exception as e:
        print(f"[s3] Failed to download model: {e}")
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
        print(f"[s3] Upload failed (non-fatal): {e}")


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
        print(f"[callback] Failed (non-fatal): {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Behavior cloning pre-training")
    p.add_argument("--job-id", required=True)
    p.add_argument("--task-type", required=True, choices=["locomotion", "reach", "grasp"])
    p.add_argument("--demo-source", required=True, choices=["experience", "expert"])
    p.add_argument("--experience-prefix", default="experience/",
                   help="S3 prefix for NDJSON experience (demo-source=experience)")
    p.add_argument("--expert-model", default="",
                   help="S3 key for expert policy (demo-source=expert)")
    p.add_argument("--demo-episodes", type=int, default=200,
                   help="Episodes to roll out from expert (demo-source=expert)")
    p.add_argument("--min-reward", type=float, default=1.0,
                   help="Minimum reward to include transition (demo-source=experience)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="/tmp/training-output")
    p.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", ""))
    p.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", "fleetos-models"))
    p.add_argument("--s3-access-key", default=os.getenv("S3_ACCESS_KEY", "fleetos"))
    p.add_argument("--s3-secret-key", default=os.getenv("S3_SECRET_KEY", "fleetos123"))
    p.add_argument("--callback-url", default=os.getenv("TRAINING_CALLBACK_URL", ""))
    args = p.parse_args()

    train(args)
