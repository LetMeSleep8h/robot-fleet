"""
Unified observation format for all FleetOS tasks.

Single source of truth for what the policy sees. Used by:
- Training environments (fleetos_env.py, fleetos_reach_env.py)
- Inference server (server.py)
- Experience replay from telemetry transitions

Observation vector (~90 dims):
  joint_pos         (25)  - all actuator angles
  joint_vel         (25)  - all actuator velocities
  root_quat         (4)   - torso orientation quaternion
  root_linvel       (3)   - torso linear velocity
  root_angvel       (3)   - torso angular velocity
  hand_pos_right    (3)   - right hand site position (world frame)
  hand_pos_left     (3)   - left hand site position (world frame)
  gripper_pos       (4)   - gripper joint angles (right_l, right_r, left_l, left_r)
  object_positions  (6)   - red_box xyz + blue_cylinder xyz
  object_velocities (6)   - red_box vel + blue_cylinder vel
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
NUM_TASKS = 4
NUM_ACTUATORS = 25
NUM_OBJECTS = 2
OBJECT_NAMES = ["red_box", "blue_cylinder"]


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

    # Root orientation and velocity
    root_quat = data.qpos[qs + 3: qs + 7].copy()  # w, x, y, z in MuJoCo
    root_linvel = data.qvel[ds: ds + 3].copy()
    root_angvel = data.qvel[ds + 3: ds + 6].copy()

    # Hand positions from sites
    right_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    left_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
    hand_pos_right = data.site_xpos[right_hand_id].copy() if right_hand_id >= 0 else np.zeros(3)
    hand_pos_left = data.site_xpos[left_hand_id].copy() if left_hand_id >= 0 else np.zeros(3)

    # Gripper joint angles (last 4 actuator joints)
    gripper_pos = joint_pos[-4:].copy()

    # Object positions and velocities
    obj_pos = np.zeros(6, dtype=np.float64)
    obj_vel = np.zeros(6, dtype=np.float64)
    for i, name in enumerate(OBJECT_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            obj_pos[i*3:(i+1)*3] = data.xpos[bid]
            if bid < len(data.cvel):
                obj_vel[i*3:(i+1)*3] = data.cvel[bid][:3]

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


def build_observation_from_transition(transition: dict, task_command: str = "") -> np.ndarray:
    """Build observation from a training transition (S3 NDJSON format).

    Used for offline RL training from collected telemetry data.
    """
    jp = np.array(transition.get("joint_pos", [0.0] * NUM_ACTUATORS), dtype=np.float32)
    jv = np.array(transition.get("joint_vel", [0.0] * NUM_ACTUATORS), dtype=np.float32)
    rp = transition.get("root_pos", [0, 0, 1.22])
    rq = np.array(transition.get("root_quat", [0, 0, 0, 1]), dtype=np.float32)
    # Derive velocities from root (not stored directly — would need consecutive frames)
    root_linvel = np.zeros(3, dtype=np.float32)
    root_angvel = np.zeros(3, dtype=np.float32)

    # Hand positions not directly in transition — approximate from joint angles
    # For proper reconstruction, we'd need forward kinematics. Use zeros as fallback.
    hand_right = np.zeros(3, dtype=np.float32)
    hand_left = np.zeros(3, dtype=np.float32)

    gripper_pos = jp[-4:].copy() if len(jp) >= NUM_ACTUATORS else np.zeros(4, dtype=np.float32)

    # Object state
    obj_pos = np.zeros(6, dtype=np.float32)
    obj_vel = np.zeros(6, dtype=np.float32)
    for i, obj in enumerate(transition.get("objects", [])[:NUM_OBJECTS]):
        obj_pos[i*3] = obj.get("pos_x", 0)
        obj_pos[i*3+1] = obj.get("pos_y", 0)
        obj_pos[i*3+2] = obj.get("pos_z", 0)
        obj_vel[i*3] = obj.get("vel_x", 0)
        obj_vel[i*3+1] = obj.get("vel_y", 0)
        obj_vel[i*3+2] = obj.get("vel_z", 0)

    cmd = task_command or transition.get("task_command", "idle")

    return build_observation(
        joint_pos=jp, joint_vel=jv,
        root_quat=rq, root_linvel=root_linvel, root_angvel=root_angvel,
        hand_pos_right=hand_right, hand_pos_left=hand_left,
        gripper_pos=gripper_pos,
        object_positions=obj_pos, object_velocities=obj_vel,
        task_command=cmd,
    )
