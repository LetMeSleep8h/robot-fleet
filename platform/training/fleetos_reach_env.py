"""
Goal-conditioned reaching environment for the FleetOS humanoid.

Uses the unified 86-dim observation format + Dict goals for HER.

Observation: Dict with "observation" (86-dim), "achieved_goal" (3), "desired_goal" (3)
Action: 25-dim position targets (joint angles in radians)
Reward: Sparse (-1 if not reached, 0 if reached) for HER compatibility
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from observation import build_observation_from_mujoco, OBS_DIM, NUM_ACTUATORS

ASSETS_DIR = Path(__file__).parent / "assets"
LAB_XML = ASSETS_DIR / "lab_room.xml"

N_SUBSTEPS = 7
DISTANCE_THRESHOLD = 0.05
FALL_Z = 0.3


class FleetOSReachEnv(gym.Env):
    """Goal-conditioned reaching: move right hand to a 3D target.

    Compatible with SB3 HerReplayBuffer via Dict observation space
    containing achieved_goal and desired_goal.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=200):
        super().__init__()

        if not LAB_XML.exists():
            raise FileNotFoundError(f"Lab XML not found: {LAB_XML}")

        self.model = mujoco.MjModel.from_xml_path(str(LAB_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # Cache site and joint IDs
        self._hand_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site"
        )
        self._root_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        self._root_qpos_adr = self.model.jnt_qposadr[self._root_jnt_id]
        self._root_dof_adr = self.model.jnt_dofadr[self._root_jnt_id]

        # Action space: position targets from XML ctrl ranges
        low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Unified 86-dim observation + Dict goals for HER
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

        self._goal = np.zeros(3)

    def _get_obs(self) -> np.ndarray:
        """Unified 86-dim observation with task_command='reach'."""
        return build_observation_from_mujoco(
            self.model, self.data,
            self._root_qpos_adr, self._root_dof_adr,
            task_command="reach",
        )

    def _get_achieved_goal(self) -> np.ndarray:
        """Current right hand position."""
        return self.data.site_xpos[self._hand_site_id].copy()

    def _sample_goal(self) -> np.ndarray:
        """Sample a random reachable 3D target near the robot.

        The workspace is roughly a sphere of radius ~0.4m centered
        at the right shoulder, offset from the robot root.
        """
        qs = self._root_qpos_adr
        robot_x = self.data.qpos[qs]
        robot_y = self.data.qpos[qs + 1]

        # Sample in a box around the robot's right side
        goal = np.array([
            robot_x + self.np_random.uniform(0.1, 0.5),
            robot_y + self.np_random.uniform(-0.4, 0.0),
            self.np_random.uniform(0.8, 1.4),  # height range
        ])
        return goal

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse reward for HER: -1 if not reached, 0 if reached."""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > DISTANCE_THRESHOLD).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Standing pose
        qs = self._root_qpos_adr
        self.data.qpos[qs + 2] = 1.22   # standing height
        self.data.qpos[qs + 3] = 1.0    # quaternion w (upright)

        # Small joint noise for exploration
        joint_start = qs + 7
        joint_end = joint_start + NUM_ACTUATORS
        noise = self.np_random.uniform(-0.02, 0.02, size=NUM_ACTUATORS)
        self.data.qpos[joint_start:joint_end] += noise

        mujoco.mj_forward(self.model, self.data)

        self._goal = self._sample_goal()
        self._step_count = 0

        obs = self._get_obs()
        return {
            "observation": obs,
            "achieved_goal": self._get_achieved_goal(),
            "desired_goal": self._goal.copy(),
        }, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:NUM_ACTUATORS] = action

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        achieved = self._get_achieved_goal()
        obs = self._get_obs()
        info = {"is_success": np.linalg.norm(achieved - self._goal) < DISTANCE_THRESHOLD}

        reward = float(self.compute_reward(achieved, self._goal, info))

        # Terminate on fall
        qs = self._root_qpos_adr
        height = self.data.qpos[qs + 2]
        terminated = height < FALL_Z
        truncated = self._step_count >= self._max_episode_steps

        return (
            {
                "observation": obs,
                "achieved_goal": achieved,
                "desired_goal": self._goal.copy(),
            },
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        pass


def register_reach_env():
    """Register the FleetOS Reach environment with Gymnasium."""
    register(
        id="FleetOS-Reach-v1",
        entry_point="fleetos_reach_env:FleetOSReachEnv",
        max_episode_steps=200,
    )
