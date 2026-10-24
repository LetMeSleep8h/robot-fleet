"""
Benchmark: online PPO (MuJoCo in the loop) vs offline CQL (fleet data only).

Runs both with the SAME number of gradient updates, so any time difference
comes from simulating the robot, not from doing less learning:

  online   PPO for --timesteps env steps on FleetOS-Humanoid-v2
           (gradient updates = rollouts x epochs x minibatches)
  offline  CQL-SAC for that many gradient steps on a synthetic dataset of
           --timesteps transitions (same shapes as fleet data, no simulator)

Prints one JSON line:
  {online_seconds, offline_seconds, online_sim_steps_per_s, offline_updates_per_s,
   cpu_count, online_gradient_updates, offline_gradient_steps, timesteps}

Usage: python benchmark_offline_vs_online.py --timesteps 20000
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

PPO_N_STEPS = 2048
PPO_BATCH = 64
PPO_EPOCHS = 10
OFFLINE_BATCH = 256


def run_online(timesteps: int) -> tuple[float, int]:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from fleetos_env import register_fleetos_env

    register_fleetos_env()
    n_steps = min(PPO_N_STEPS, timesteps)
    batch = min(PPO_BATCH, n_steps)
    env = DummyVecEnv([lambda: gym.make("FleetOS-Humanoid-v2")])
    model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch, n_epochs=PPO_EPOCHS, verbose=0)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    seconds = time.perf_counter() - start
    env.close()
    rollouts = math.ceil(timesteps / n_steps)
    return seconds, rollouts * PPO_EPOCHS * (n_steps // batch)


def run_offline(transitions: int, gradient_steps: int) -> float:
    import train_offline as to

    rng = np.random.default_rng(0)
    data = {
        "observations": rng.normal(size=(transitions, to.OBS_DIM)).astype(np.float32),
        "next_observations": rng.normal(size=(transitions, to.OBS_DIM)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(transitions, to.ACT_DIM)).astype(np.float32),
        "rewards": rng.normal(size=transitions).astype(np.float32),
        "dones": (rng.random(transitions) < 0.01).astype(np.float32),
    }
    args = to.build_parser().parse_args([
        "--job-id", "bench", "--dataset-manifest", "-", "--batch-size", str(min(OFFLINE_BATCH, transitions)),
    ])
    model = to.make_model(args, buffer_size=transitions)
    to.fill_replay_buffer(model, data)
    start = time.perf_counter()
    to.CQLTrainer(model).train_offline(gradient_steps, batch_size=args.batch_size,
                                       log_interval=max(gradient_steps, 1))
    return time.perf_counter() - start


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--timesteps", type=int, default=20_000)
    args = p.parse_args(argv)

    online_s, updates = run_online(args.timesteps)
    offline_s = run_offline(args.timesteps, updates)
    print(json.dumps({
        "timesteps": args.timesteps,
        "online_seconds": round(online_s, 2),
        "offline_seconds": round(offline_s, 2),
        "online_sim_steps_per_s": round(args.timesteps / online_s, 1),
        "offline_updates_per_s": round(updates / offline_s, 1),
        "online_gradient_updates": updates,
        "offline_gradient_steps": updates,
        "cpu_count": os.cpu_count(),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
