"""
Unified observation format for all FleetOS tasks.

Single source of truth for what the policy sees. Used by:
- Training environments (fleetos_env.py, fleetos_reach_env.py)
- Inference server (server.py)
- Experience replay from telemetry transitions

Observation vector, version 3 (egocentric). Everything is expressed relative
to the robot: positions from the torso, vectors in the robot's heading frame
(world z-rotation removed), so the same state reads the same wherever the
robot stands and whichever way it faces.
  joint_pos         (25)  - all actuator angles
  joint_vel         (25)  - all actuator velocities
  root_quat         (4)   - torso orientation with heading removed (w, x, y, z)
  root_linvel       (3)   - torso linear velocity (heading frame)
  root_angvel       (3)   - torso angular velocity (body frame)
  hand_pos_right    (3)   - right hand position relative to the torso (heading frame)
  hand_pos_left     (3)   - left hand position relative to the torso (heading frame)
  gripper_pos       (4)   - gripper joint angles (right_l, right_r, left_l, left_r)
  object_positions  (6)   - red_box + blue_cylinder positions relative to the torso (heading frame)
  object_velocities (6)   - red_box + blue_cylinder linear velocities (heading frame)
  task_encoding     (4)   - one-hot: [locomotion, reach, grasp, idle]
  ─────────────────────
  Total:            86

Goal (separate Dict field for HER):
  desired_goal      (3)   - target position
  achieved_goal     (3)   - current hand position
"""

import numpy as np

# Task IDs for one-hot encoding
TASK_IDS = {
    "locomotion": 0,
    "reach": 1,
    "grasp": 2,
    "idle": 3,
}

# Task command → task category mapping
COMMAND_TO_TASK = {
    "move": "locomotion",
    "move_relative": "locomotion",
    "walk": "locomotion",
    "reach": "reach",
    "pick_up": "grasp",
    "grasp": "grasp",
    "release": "grasp",
    "idle": "idle",
    "stop": "idle",
    "wave": "idle",
    "dance": "idle",
    "bow": "idle",
    "sit": "idle",
    "jump": "idle",
    "look_around": "idle",
    "stretch": "idle",
}

OBS_DIM = 86
# Bump when the meaning of any observation slot changes; policies trained on a
# different version must not be executed (see policy_meta.json).
OBS_VERSION = 3
NUM_TASKS = 4
NUM_ACTUATORS = 25
NUM_OBJECTS = 2
OBJECT_NAMES = ["red_box", "blue_cylinder"]

# Slot layout of the observation vector.
JOINT_POS_SLICE = slice(0, 25)
JOINT_VEL_SLICE = slice(25, 50)
ROOT_QUAT_SLICE = slice(50, 54)
ROOT_LINVEL_SLICE = slice(54, 57)
ROOT_ANGVEL_SLICE = slice(57, 60)
HAND_SLICE = slice(60, 66)
GRIPPER_SLICE = slice(66, 70)
OBJECT_POS_SLICE = slice(70, 76)
OBJECT_VEL_SLICE = slice(76, 82)
TASK_SLICE = slice(82, 86)


def heading_yaw(quat_wxyz: np.ndarray) -> float:
    """Rotation about world z (heading) of a (w, x, y, z) quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def to_heading_frame(vec: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world-frame vector by -yaw so +x is the robot's facing direction."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1], vec[2]])


def remove_heading(quat_wxyz: np.ndarray, yaw: float) -> np.ndarray:
    """Orientation with the heading (yaw) removed: q_yaw^-1 * q, as (w, x, y, z)."""
    hw, hz = np.cos(yaw / 2.0), -np.sin(yaw / 2.0)  # inverse yaw quaternion (hw, 0, 0, hz)
    w, x, y, z = quat_wxyz
    return np.array([
        hw * w - hz * z,
        hw * x - hz * y,
        hw * y + hz * x,
        hw * z + hz * w,
    ])


def build_observation(
    joint_pos: np.ndarray,       # (25,)
    joint_vel: np.ndarray,       # (25,)
    root_quat: np.ndarray,       # (4,) - quaternion (x, y, z, w)
    root_linvel: np.ndarray,     # (3,)
    root_angvel: np.ndarray,     # (3,)
    hand_pos_right: np.ndarray,  # (3,)
    hand_pos_left: np.ndarray,   # (3,)
    gripper_pos: np.ndarray,     # (4,) - right_l, right_r, left_l, left_r
    object_positions: np.ndarray,  # (6,) - [rx, ry, rz, bx, by, bz]
    object_velocities: np.ndarray, # (6,)
    task_command: str = "idle",
) -> np.ndarray:
    """Build the unified observation vector (86 dims).

    All inputs are numpy arrays. Returns float32.
    """
    task_id = TASK_IDS.get(COMMAND_TO_TASK.get(task_command, "idle"), 3)
    task_onehot = np.zeros(NUM_TASKS, dtype=np.float32)
    task_onehot[task_id] = 1.0

    obs = np.concatenate([
        joint_pos.astype(np.float32),
        joint_vel.astype(np.float32),
        root_quat.astype(np.float32),
        root_linvel.astype(np.float32),
        root_angvel.astype(np.float32),
        hand_pos_right.astype(np.float32),
        hand_pos_left.astype(np.float32),
        gripper_pos.astype(np.float32),
        object_positions.astype(np.float32),
        object_velocities.astype(np.float32),
        task_onehot,
    ])
    assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM} dims, got {obs.shape[0]}"
    return obs


def build_observation_from_mujoco(model, data, root_qpos_adr: int, root_dof_adr: int,
                                  task_command: str = "idle") -> np.ndarray:
    """Build observation directly from MuJoCo model/data.

    Args:
        model: MuJoCo MjModel
        data: MuJoCo MjData
        root_qpos_adr: qpos address of the robot's root joint
        root_dof_adr: dof address of the robot's root joint
        task_command: current task/command name
    """
    import mujoco

    qs = root_qpos_adr
    ds = root_dof_adr

    # Joint positions and velocities (25 actuator joints, after 7 root DOFs)
    joint_pos = data.qpos[qs + 7: qs + 7 + NUM_ACTUATORS].copy()
    joint_vel = data.qvel[ds + 6: ds + 6 + NUM_ACTUATORS].copy()

    # Root: everything relative to the torso and its heading
    root_pos = data.qpos[qs: qs + 3].copy()
    world_quat = data.qpos[qs + 3: qs + 7].copy()  # w, x, y, z in MuJoCo
    yaw = heading_yaw(world_quat)
    root_quat = remove_heading(world_quat, yaw)
    root_linvel = to_heading_frame(data.qvel[ds: ds + 3], yaw)  # free-joint linear vel is world-frame
    root_angvel = data.qvel[ds + 3: ds + 6].copy()              # free-joint angular vel is body-frame

    def relative(point: np.ndarray) -> np.ndarray:
        return to_heading_frame(point - root_pos, yaw)

    # Hand positions from sites, relative to the torso
    right_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    left_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
    hand_pos_right = relative(data.site_xpos[right_hand_id]) if right_hand_id >= 0 else np.zeros(3)
    hand_pos_left = relative(data.site_xpos[left_hand_id]) if left_hand_id >= 0 else np.zeros(3)

    # Gripper joint angles (last 4 actuator joints)
    gripper_pos = joint_pos[-4:].copy()

    # Object positions (relative) and linear velocities (heading frame)
    obj_pos = np.zeros(6, dtype=np.float64)
    obj_vel = np.zeros(6, dtype=np.float64)
    vel6 = np.zeros(6)
    for i, name in enumerate(OBJECT_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            obj_pos[i*3:(i+1)*3] = relative(data.xpos[bid])
            # [angular(3), linear(3)] in the world frame (flag 0)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            obj_vel[i*3:(i+1)*3] = to_heading_frame(vel6[3:], yaw)

    return build_observation(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_quat=root_quat,
        root_linvel=root_linvel,
        root_angvel=root_angvel,
        hand_pos_right=hand_pos_right,
        hand_pos_left=hand_pos_left,
        gripper_pos=gripper_pos,
        object_positions=obj_pos,
        object_velocities=obj_vel,
        task_command=task_command,
    )
