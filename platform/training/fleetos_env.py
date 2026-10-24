"""
Custom Gymnasium environment wrapping our lab_room.xml MuJoCo model.

Uses the unified 86-dim observation format (observation.py) and
task-conditional reward matching the simulator.

Registers as "FleetOS-Humanoid-v2" in Gymnasium.
"""

import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from observation import build_observation_from_mujoco, OBS_DIM, NUM_ACTUATORS
from locomotion_reward import MAX_EPISODE_STEPS, control_cost, has_fallen, locomotion_reward

ASSETS_DIR = Path(__file__).parent / "assets"
LAB_XML = ASSETS_DIR / "lab_room.xml"

N_SUBSTEPS = 7  # 0.003s × 7 = 0.021s per step


class FleetOSHumanoidEnv(gym.Env):
    """Gymnasium env using our lab_room.xml with Humanoid-v4 compatible API."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=MAX_EPISODE_STEPS):
        super().__init__()

        if not LAB_XML.exists():
            raise FileNotFoundError(f"Lab XML not found: {LAB_XML}")

        self.model = mujoco.MjModel.from_xml_path(str(LAB_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # Find robot root joint (after dynamic object freejoints)
        self._root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos_adr = self.model.jnt_qposadr[self._root_jnt_id]
        self._root_dof_adr = self.model.jnt_dofadr[self._root_jnt_id]

        # Action space: position targets for each actuator (from XML ctrl ranges)
        low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Unified 86-dim observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._prev_x = 0.0

    def _get_obs(self) -> np.ndarray:
        """Build unified 86-dim observation from MuJoCo state."""
        return build_observation_from_mujoco(
            self.model, self.data,
            self._root_qpos_adr, self._root_dof_adr,
            task_command="walk",
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qs = self._root_qpos_adr
        self.data.qpos[qs + 2] = 1.22  # standing height
        self.data.qpos[qs + 3] = 1.0   # quaternion w (upright)

        # Small noise on joints only
        joint_start = qs + 7
        joint_end = joint_start + NUM_ACTUATORS
        noise = self.np_random.uniform(-0.01, 0.01, size=NUM_ACTUATORS)
        self.data.qpos[joint_start:joint_end] += noise

        mujoco.mj_forward(self.model, self.data)

        self._prev_x = self.data.qpos[qs]
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:NUM_ACTUATORS] = action

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        qs = self._root_qpos_adr
        height = self.data.qpos[qs + 2]
        x_pos = self.data.qpos[qs]

        # Locomotion reward (matches simulator task-conditional reward for "walk")
        dt = self.model.opt.timestep * N_SUBSTEPS
        forward_velocity = (x_pos - self._prev_x) / dt
        self._prev_x = x_pos

        ctrl_cost = control_cost(action)
        # Shared with the robots (locomotion_reward.py) so the test track
        # scores and terminates exactly like the fleet.
        reward = locomotion_reward(forward_velocity, height, ctrl_cost)
        terminated = has_fallen(height)
        truncated = self._step_count >= self._max_episode_steps

        info = {
            "reward_forward": forward_velocity,
            "reward_ctrl": -ctrl_cost,
            "height": height,
            "x_position": x_pos,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        pass


def register_fleetos_env():
    """Register the FleetOS Humanoid environment with Gymnasium."""
    register(
        id="FleetOS-Humanoid-v2",
        entry_point="fleetos_env:FleetOSHumanoidEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
