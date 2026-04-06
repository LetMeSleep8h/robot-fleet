"""
Goal-conditioned grasping environment for the FleetOS humanoid.

The robot must reach an object (red_box or blue_cylinder on the workbench),
close its gripper around it, and lift it to a target height.

Uses Dict observation space for HER compatibility:
  observation: 86-dim unified format (task_encoding[2]=1 for grasp)
  achieved_goal: 3D object position
  desired_goal: 3D lift target (object initial pos + lift offset)

Reward: Sparse (-1 if not at goal, 0 if at goal) for HER.
Success: object within 0.05m of goal AND gripper closed.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from observation import (
    build_observation_from_mujoco, OBS_DIM, NUM_ACTUATORS, OBJECT_NAMES,
)

ASSETS_DIR = Path(__file__).parent / "assets"
LAB_XML = ASSETS_DIR / "lab_room.xml"

N_SUBSTEPS = 7
DISTANCE_THRESHOLD = 0.05  # 5cm for object-at-goal
LIFT_OFFSET = 0.15         # lift 15cm above table surface
FALL_Z = 0.3
GRIPPER_CLOSED_THRESHOLD = 0.3  # gripper joint angle (rad) considered closed


class FleetOSGraspEnv(gym.Env):
    """Goal-conditioned grasping: pick up an object and lift it.

    Compatible with SB3 HerReplayBuffer via Dict observation space.

    The achieved_goal tracks the object position (not the hand), and the
    desired_goal is the lift target above the object's starting position.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=300):
        super().__init__()

        if not LAB_XML.exists():
            raise FileNotFoundError(f"Lab XML not found: {LAB_XML}")

        self.model = mujoco.MjModel.from_xml_path(str(LAB_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # Cache IDs
        self._hand_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site"
        )
        self._root_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        self._root_qpos_adr = self.model.jnt_qposadr[self._root_jnt_id]
        self._root_dof_adr = self.model.jnt_dofadr[self._root_jnt_id]

        # Object body IDs
        self._object_body_ids = {}
        self._object_joint_ids = {}
        for name in OBJECT_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )
            if bid >= 0:
                self._object_body_ids[name] = bid
            if jid >= 0:
                self._object_joint_ids[name] = jid

        # Gripper actuator indices (last 4 actuators: right_l, right_r, left_l, left_r)
        # We use right gripper (indices -4 and -3 relative to actuator list)
        self._right_gripper_indices = [NUM_ACTUATORS - 4, NUM_ACTUATORS - 3]

        # Action space: position targets from XML ctrl ranges
        low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Dict observation space for HER
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
        })

        self._goal = np.zeros(3)
        self._target_object = "red_box"
        self._object_initial_pos = np.zeros(3)

    def _get_obs(self) -> np.ndarray:
        """Unified 86-dim observation with task_command='grasp'."""
        return build_observation_from_mujoco(
            self.model, self.data,
            self._root_qpos_adr, self._root_dof_adr,
            task_command="grasp",
        )

    def _get_object_pos(self) -> np.ndarray:
        """Current position of the target object."""
        bid = self._object_body_ids.get(self._target_object)
        if bid is not None and bid >= 0:
            return self.data.xpos[bid].copy()
        return np.zeros(3)

    def _get_achieved_goal(self) -> np.ndarray:
        """Current object position (what we're trying to move to the goal)."""
        return self._get_object_pos()

    def _is_gripper_closed(self) -> bool:
        """Check if right gripper fingers are closed around something."""
        qs = self._root_qpos_adr
        joint_start = qs + 7
        # Right gripper joint angles (actuator indices map to joints)
        gripper_l = self.data.qpos[joint_start + self._right_gripper_indices[0]]
        gripper_r = self.data.qpos[joint_start + self._right_gripper_indices[1]]
        return (gripper_l > GRIPPER_CLOSED_THRESHOLD and
                gripper_r > GRIPPER_CLOSED_THRESHOLD)

    def _sample_goal(self) -> np.ndarray:
        """Lift target: object start position + vertical offset."""
        lift_height = self.np_random.uniform(0.10, 0.20)
        goal = self._object_initial_pos.copy()
        goal[2] += lift_height
        return goal

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse reward for HER: -1 if not at goal, 0 if at goal.

        For batched HER relabeling, info may not contain gripper state,
        so we use distance-only reward. The gripper check is applied
        in the is_success flag during step().
        """
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > DISTANCE_THRESHOLD).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Standing pose near the workbench
        qs = self._root_qpos_adr
        self.data.qpos[qs + 0] = 2.5    # x: near workbench at x=3
        self.data.qpos[qs + 1] = 3.0    # y: centered on workbench
        self.data.qpos[qs + 2] = 1.22   # z: standing height
        self.data.qpos[qs + 3] = 1.0    # quaternion w (upright)

        # Small joint noise
        joint_start = qs + 7
        joint_end = joint_start + NUM_ACTUATORS
        noise = self.np_random.uniform(-0.02, 0.02, size=NUM_ACTUATORS)
        self.data.qpos[joint_start:joint_end] += noise

        mujoco.mj_forward(self.model, self.data)

        # Randomly pick target object
        self._target_object = self.np_random.choice(OBJECT_NAMES)
        self._object_initial_pos = self._get_object_pos()
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

        obj_at_goal = np.linalg.norm(achieved - self._goal) < DISTANCE_THRESHOLD
        gripper_closed = self._is_gripper_closed()
        is_success = bool(obj_at_goal and gripper_closed)

        info = {
            "is_success": is_success,
            "object_at_goal": bool(obj_at_goal),
            "gripper_closed": gripper_closed,
            "object_distance": float(np.linalg.norm(achieved - self._goal)),
            "target_object": self._target_object,
        }

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


def register_grasp_env():
    """Register the FleetOS Grasp environment with Gymnasium."""
    register(
        id="FleetOS-Grasp-v1",
        entry_point="fleetos_grasp_env:FleetOSGraspEnv",
        max_episode_steps=300,
    )
