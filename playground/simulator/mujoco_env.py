"""
MuJoCo physics simulation with custom lab room environment.

Uses the direct mujoco Python API (not gymnasium) for full control
over the scene and dynamic robot spawning. Rendering is done
client-side via Three.js.
"""

import logging
import os
from pathlib import Path

import mujoco
import numpy as np

from locomotion_reward import FALL_PENALTY, control_cost, has_fallen, locomotion_reward

from joint_mapping import extract_joint_states, extract_root_pose, MUJOCO_QPOS_OFFSET

log = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"
LAB_XML = ASSETS_DIR / "lab_room.xml"

# Humanoid body constants (17 body + 4 ankle + 4 gripper = 25 actuators)
NUM_ACTUATORS = 25
ROOT_QPOS_SIZE = 7   # 3 pos + 4 quat
ROOT_QVEL_SIZE = 6   # 3 lin vel + 3 ang vel
JOINT_QPOS_SIZE = 25
JOINT_QVEL_SIZE = 25
BODY_QPOS_SIZE = ROOT_QPOS_SIZE + JOINT_QPOS_SIZE  # 32
BODY_QVEL_SIZE = ROOT_QVEL_SIZE + JOINT_QVEL_SIZE   # 31



class LabSimulation:
    """Manages the MuJoCo lab room with one or more humanoid robots."""

    def __init__(self):
        self._base_xml = LAB_XML.read_text()
        self.robot_count = 1
        self.robots = []
        self.steps = 0
        self._load_model(self._build_xml(1))

    def _load_model(self, xml_string: str):
        """Load or reload the MuJoCo model from an XML string."""
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Rebuild robot tracking using mj_name2id for correct offsets.
        # Dynamic objects (red_box, blue_cylinder) have freejoints that
        # occupy qpos slots before the humanoid, so we can't assume
        # the humanoid starts at qpos[0].
        self.robots = []
        for i in range(self.robot_count):
            suffix = "" if i == 0 else f"_{i+1}"
            torso_name = f"torso{suffix}"

            # Find the root joint for this robot
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"root{suffix}")
            qpos_start = self.model.jnt_qposadr[jnt_id]
            qvel_start = self.model.jnt_dofadr[jnt_id]

            # Find the first actuator for this robot
            act_name = f"abdomen_y{suffix}"
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            actuator_start = act_id  # actuators are contiguous per robot

            self.robots.append({
                "id": f"robot-{i+1:04d}",
                "qpos_start": qpos_start,
                "qvel_start": qvel_start,
                "dof_start": qvel_start,   # same as qvel_start, used by observation builder
                "actuator_start": actuator_start,
                "battery": 1.0,
                "status": "active",
                # Start from the spawn position: prev_x=0 credited a huge fake
                # forward velocity on the first step after every spawn/reload.
                "prev_x": float(self.data.qpos[qpos_start]),
                "reward_accum": 0.0,
                "episode_steps": 0,
                "fall_count": 0,
                "episode_count": 0,
                "total_reward": 0.0,
                "total_steps": 0,
                "cum_reward": 0.0,  # monotonic (never reset); transition rewards are differences
            })

            log.info("Robot %s: qpos_start=%d, qvel_start=%d, act_start=%d",
                     self.robots[-1]["id"], qpos_start, qvel_start, actuator_start)

        mujoco.mj_step(self.model, self.data)
        for robot in self.robots:  # re-sync after the settling step
            robot["prev_x"] = float(self.data.qpos[robot["qpos_start"]])
        log.info("Model loaded: %d robots, %d actuators, %d qpos",
                 len(self.robots), self.model.nu, self.model.nq)

    def _build_xml(self, count: int) -> str:
        """Build XML string with `count` humanoid bodies.

        The base XML has 1 humanoid. For additional robots, we duplicate
        the humanoid body/tendon/actuator blocks with unique names and
        different spawn positions.
        """
        if count <= 1:
            return self._base_xml

        import re

        xml = self._base_xml
        # Extract the humanoid body block (between torso body and </worldbody>)
        body_match = re.search(
            r'(<!-- ===== Humanoid Robot.*?)(<!-- ===== Cameras)',
            xml, re.DOTALL
        )
        if not body_match:
            # Fallback: find the torso body directly
            body_match = re.search(
                r'(<body name="torso".*?</body>\s*</body>\s*</body>\s*</body>\s*</body>)',
                xml, re.DOTALL
            )

        tendon_match = re.search(r'(<tendon>.*?</tendon>)', xml, re.DOTALL)
        actuator_match = re.search(r'(<actuator>.*?</actuator>)', xml, re.DOTALL)

        if not tendon_match or not actuator_match:
            log.error("Could not parse XML for multi-robot spawn")
            return xml

        tendon_block = tendon_match.group(1)
        actuator_block = actuator_match.group(1)

        # Spawn positions: spread around the room
        positions = [
            (0, 0),      # robot 1 (already in XML)
            (2, 2),      # robot 2
            (-2, 2),     # robot 3
            (2, -2),     # robot 4
            (-2, -2),    # robot 5
            (0, 3),      # robot 6
            (3, 0),      # robot 7
            (-3, 0),     # robot 8
        ]

        extra_bodies = ""
        extra_tendons = ""
        extra_actuators = ""

        for i in range(1, count):
            suffix = f"_{i+1}"
            px, py = positions[i] if i < len(positions) else (i * 1.5 % 8 - 4, i * 1.3 % 8 - 4)

            # Duplicate the torso body with renamed joints/geoms and new position
            body = self._make_humanoid_body(suffix, px, py)
            extra_bodies += body + "\n"

            # Duplicate tendons
            extra_tendons += f"""
    <fixed name="left_hipknee{suffix}">
        <joint coef="-1" joint="left_hip_y{suffix}"/>
        <joint coef="1" joint="left_knee{suffix}"/>
    </fixed>
    <fixed name="right_hipknee{suffix}">
        <joint coef="-1" joint="right_hip_y{suffix}"/>
        <joint coef="1" joint="right_knee{suffix}"/>
    </fixed>"""

            # Duplicate actuators
            actuator_names = [
                "abdomen_y", "abdomen_z", "abdomen_x",
                "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
                "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
                "right_shoulder1", "right_shoulder2", "right_elbow",
                "left_shoulder1", "left_shoulder2", "left_elbow",
                "right_ankle_y", "right_ankle_x", "left_ankle_y", "left_ankle_x",
                "right_gripper_l", "right_gripper_r", "left_gripper_l", "left_gripper_r",
            ]
            kps = [200, 200, 200, 300, 200, 400, 350, 300, 200, 400, 350, 80, 80, 60, 80, 80, 60,
                   300, 150, 300, 150, 20, 20, 20, 20]
            for name, kp in zip(actuator_names, kps):
                extra_actuators += f'        <position joint="{name}{suffix}" name="{name}{suffix}" kp="{kp}"/>\n'

        # Insert extra bodies before cameras
        xml = xml.replace(
            "<!-- ===== Cameras =====",
            extra_bodies + "\n        <!-- ===== Cameras ====="
        )

        # Insert extra tendons
        xml = xml.replace("</tendon>", extra_tendons + "\n    </tendon>")

        # Insert extra actuators
        xml = xml.replace("</actuator>", extra_actuators + "    </actuator>")

        return xml

    def _make_humanoid_body(self, suffix: str, px: float, py: float) -> str:
        """Generate a humanoid body block with renamed joints and position offset."""
        # This is the humanoid body from the base XML with all names suffixed
        return f"""
        <body name="torso{suffix}" pos="{px} {py} 1.22">
            <joint armature="0" damping="0" limited="false" name="root{suffix}" pos="0 0 0" stiffness="0" type="free"/>
            <geom fromto="0 -.07 0 0 .07 0" name="torso1{suffix}" size="0.07" type="capsule"/>
            <geom name="head{suffix}" pos="0 0 .19" size=".09" type="sphere" user="258"/>
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist{suffix}" size="0.06" type="capsule"/>
            <body name="lwaist{suffix}" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="lwaist{suffix}" size="0.06" type="capsule"/>
                <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z{suffix}" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y{suffix}" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="pelvis{suffix}" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x{suffix}" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt{suffix}" size="0.09" type="capsule"/>
                    <body name="right_thigh{suffix}" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x{suffix}" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z{suffix}" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y{suffix}" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1{suffix}" size="0.06" type="capsule"/>
                        <body name="right_shin{suffix}" pos="0 0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="right_knee{suffix}" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="right_shin1{suffix}" size="0.049" type="capsule"/>
                            <body name="right_foot{suffix}" pos="0 0 -0.35">
                                <joint armature="0.01" axis="0 1 0" damping="8" name="right_ankle_y{suffix}" pos="0 0 0.05" range="-45 45" stiffness="20" type="hinge"/>
                                <joint armature="0.008" axis="1 0 0" damping="5" name="right_ankle_x{suffix}" pos="0 0 0.05" range="-25 25" stiffness="15" type="hinge"/>
                                <geom name="right_foot{suffix}" type="box" size="0.1 0.05 0.015" pos="0.03 0 0" condim="4" contype="1" conaffinity="1" friction="1.2 0.005 0.001" rgba="0.5 0.5 0.5 1"/>
                            </body>
                        </body>
                    </body>
                    <body name="left_thigh{suffix}" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x{suffix}" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z{suffix}" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y{suffix}" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1{suffix}" size="0.06" type="capsule"/>
                        <body name="left_shin{suffix}" pos="0 -0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="left_knee{suffix}" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="left_shin1{suffix}" size="0.049" type="capsule"/>
                            <body name="left_foot{suffix}" pos="0 0 -0.35">
                                <joint armature="0.01" axis="0 1 0" damping="8" name="left_ankle_y{suffix}" pos="0 0 0.05" range="-45 45" stiffness="20" type="hinge"/>
                                <joint armature="0.008" axis="1 0 0" damping="5" name="left_ankle_x{suffix}" pos="0 0 0.05" range="-25 25" stiffness="15" type="hinge"/>
                                <geom name="left_foot{suffix}" type="box" size="0.1 0.05 0.015" pos="0.03 0 0" condim="4" contype="1" conaffinity="1" friction="1.2 0.005 0.001" rgba="0.5 0.5 0.5 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="right_upper_arm{suffix}" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="right_shoulder1{suffix}" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2{suffix}" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1{suffix}" size="0.04 0.16" type="capsule"/>
                <body name="right_lower_arm{suffix}" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="right_elbow{suffix}" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm{suffix}" size="0.031" type="capsule"/>
                    <body name="right_gripper_base{suffix}" pos=".18 .18 .18">
                        <site name="right_hand_site{suffix}" size="0.01" rgba="1 0 0 0.5"/>
                        <geom name="right_palm{suffix}" type="box" size="0.025 0.03 0.015" mass="0.1" rgba="0.4 0.5 0.6 1"/>
                        <body name="right_finger_l{suffix}" pos="0 -0.03 0.015">
                            <joint name="right_gripper_l{suffix}" type="hinge" axis="0 -1 0" range="0 45" damping="0.5" armature="0.005" stiffness="0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.008" condim="4" contype="1" conaffinity="1" friction="1.5 0.005 0.001" rgba="0.5 0.6 0.7 1"/>
                        </body>
                        <body name="right_finger_r{suffix}" pos="0 0.03 0.015">
                            <joint name="right_gripper_r{suffix}" type="hinge" axis="0 1 0" range="0 45" damping="0.5" armature="0.005" stiffness="0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.008" condim="4" contype="1" conaffinity="1" friction="1.5 0.005 0.001" rgba="0.5 0.6 0.7 1"/>
                        </body>
                    </body>
                </body>
            </body>
            <body name="left_upper_arm{suffix}" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1{suffix}" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="left_shoulder2{suffix}" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1{suffix}" size="0.04 0.16" type="capsule"/>
                <body name="left_lower_arm{suffix}" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="left_elbow{suffix}" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm{suffix}" size="0.031" type="capsule"/>
                    <body name="left_gripper_base{suffix}" pos=".18 -.18 .18">
                        <site name="left_hand_site{suffix}" size="0.01" rgba="1 0 0 0.5"/>
                        <geom name="left_palm{suffix}" type="box" size="0.025 0.03 0.015" mass="0.1" rgba="0.4 0.5 0.6 1"/>
                        <body name="left_finger_l{suffix}" pos="0 0.03 0.015">
                            <joint name="left_gripper_l{suffix}" type="hinge" axis="0 1 0" range="0 45" damping="0.5" armature="0.005" stiffness="0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.008" condim="4" contype="1" conaffinity="1" friction="1.5 0.005 0.001" rgba="0.5 0.6 0.7 1"/>
                        </body>
                        <body name="left_finger_r{suffix}" pos="0 -0.03 0.015">
                            <joint name="left_gripper_r{suffix}" type="hinge" axis="0 -1 0" range="0 45" damping="0.5" armature="0.005" stiffness="0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.008" condim="4" contype="1" conaffinity="1" friction="1.5 0.005 0.001" rgba="0.5 0.6 0.7 1"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>"""

    def spawn_robot(self) -> dict:
        """Add a new humanoid to the scene. Rebuilds the MuJoCo model."""
        self.robot_count += 1
        new_xml = self._build_xml(self.robot_count)
        self._load_model(new_xml)
        new_robot = self.robots[-1]
        log.info("Spawned %s (total: %d)", new_robot["id"], self.robot_count)
        return {"robot_id": new_robot["id"], "total_robots": self.robot_count}

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds."""
        return self.data.time

    def action_bounds(self, robot_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-actuator (low, high) control range for a robot, from the scene."""
        robot = self._find_robot(robot_id)
        start = robot["actuator_start"]
        ranges = self.model.actuator_ctrlrange[start:start + NUM_ACTUATORS]
        return ranges[:, 0], ranges[:, 1]

    def step(self, actions: dict[str, np.ndarray], task_commands: dict[str, str] | None = None,
             n_substeps: int = 7):
        """Advance physics by n_substeps to match real-time at 50 Hz control rate.

        Args:
            actions: dict mapping robot_id -> 25-dim action array
            task_commands: dict mapping robot_id -> current command name (for task-conditional reward)
            n_substeps: number of mj_step calls per control tick (default: 7)
        """
        if task_commands is None:
            task_commands = {}

        # Apply actions to actuators (held constant across sub-steps), clipped
        # to each actuator's control range exactly like the training env.
        for robot in self.robots:
            rid = robot["id"]
            if rid in actions:
                low, high = self.action_bounds(rid)
                start = robot["actuator_start"]
                self.data.ctrl[start:start + NUM_ACTUATORS] = np.clip(actions[rid], low, high)

        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        dt = self.model.opt.timestep * n_substeps

        # Compute task-conditional reward and check for falls
        for robot in self.robots:
            qs = robot["qpos_start"]
            vs = robot["qvel_start"]
            acts = robot["actuator_start"]

            height = self.data.qpos[qs + 2]
            x_pos = self.data.qpos[qs]
            cmd = task_commands.get(robot["id"], "idle")

            ctrl = self.data.ctrl[acts:acts + NUM_ACTUATORS]
            ctrl_cost = control_cost(ctrl)

            reward = self._compute_reward(robot, cmd, height, x_pos, dt, ctrl_cost)
            robot["prev_x"] = x_pos
            robot["reward_accum"] += reward
            robot["cum_reward"] += reward
            robot["episode_steps"] += 1
            robot["total_steps"] += 1

            # Fall detection and reset (same rule as the training env)
            if has_fallen(height):
                robot["fall_count"] += 1
                robot["episode_count"] += 1
                robot["total_reward"] += robot["reward_accum"]
                robot["reward_accum"] = 0.0
                robot["episode_steps"] = 0
                self._reset_robot(robot)

            # Battery model
            robot["battery"] = max(0.0, robot["battery"] - 0.000005)
            if robot["battery"] < 0.1:
                robot["status"] = "charging"
                robot["battery"] = min(1.0, robot["battery"] + 0.0003)
            elif robot["battery"] > 0.95:
                robot["status"] = "active"

    def _compute_reward(self, robot: dict, cmd: str, height: float,
                        x_pos: float, dt: float, ctrl_cost: float) -> float:
        """Task-conditional reward.

        Different commands get different reward signals:
        - idle/stop: small penalty per step (no reward for doing nothing)
        - walk/move: forward velocity reward
        - reach/pick_up/grasp: negative distance from hand to target object
        - wave/dance/bow/etc: small reward for staying upright during gesture
        """
        # Penalty for falling applies to all tasks
        if has_fallen(height):
            return FALL_PENALTY

        if cmd in ("move", "move_relative", "walk"):
            # Locomotion: identical to the training env / test track
            forward_vel = (x_pos - robot["prev_x"]) / dt if dt > 0 else 0.0
            return locomotion_reward(forward_vel, height, ctrl_cost)

        if cmd in ("reach", "pick_up", "grasp"):
            # Manipulation: reward hand-object proximity
            hand_dist = self._hand_object_distance(robot)
            if hand_dist is not None:
                return -hand_dist - 0.1 * ctrl_cost  # closer = better
            return -1.0 - 0.1 * ctrl_cost

        if cmd in ("wave", "dance", "bow", "sit", "jump", "stretch", "look_around"):
            # Gesture: small reward for staying upright, no free points
            upright = 1.0 if height > 0.8 else 0.0
            return upright * 0.5 - ctrl_cost

        if cmd in ("idle", "stop"):
            # Idle: no reward. Robot should not be rewarded for standing still.
            return -0.01 - 0.1 * ctrl_cost

        # Unknown command: small negative to discourage doing nothing
        return -0.01 - ctrl_cost

    def _hand_object_distance(self, robot: dict) -> float | None:
        """Distance from right hand site to nearest dynamic object site."""
        try:
            hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
            hand_pos = self.data.site_xpos[hand_id]

            min_dist = float("inf")
            for obj_name in ("red_box_site", "blue_cylinder_site"):
                obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, obj_name)
                if obj_id >= 0:
                    obj_pos = self.data.site_xpos[obj_id]
                    dist = float(np.linalg.norm(hand_pos - obj_pos))
                    min_dist = min(min_dist, dist)

            return min_dist if min_dist < float("inf") else None
        except Exception:
            return None

    def _reset_robot(self, robot: dict):
        """Reset a single robot to standing pose."""
        qpos_start = robot["qpos_start"]
        qvel_start = robot["qvel_start"]

        # Standing pose: position at (x, y, 1.4), upright quaternion
        current_x = self.data.qpos[qpos_start]
        current_y = self.data.qpos[qpos_start + 1]
        self.data.qpos[qpos_start:qpos_start + BODY_QPOS_SIZE] = 0.0
        self.data.qpos[qpos_start + 0] = current_x  # keep x
        self.data.qpos[qpos_start + 1] = current_y  # keep y
        self.data.qpos[qpos_start + 2] = 1.22        # standing height (feet at floor)
        self.data.qpos[qpos_start + 3] = 1.0         # quaternion w
        self.data.qvel[qvel_start:qvel_start + BODY_QVEL_SIZE] = 0.0

        if self.steps % 500 == 0:
            log.info("Reset %s at step %d", robot["id"], self.steps)

    def get_robot_state(self, robot_id: str) -> dict | None:
        """Extract state for a specific robot."""
        robot = self._find_robot(robot_id)
        if not robot:
            return None

        qs = robot["qpos_start"]
        vs = robot["qvel_start"]
        acts = robot["actuator_start"]

        qpos = self.data.qpos[qs:qs + BODY_QPOS_SIZE].copy()
        qvel = self.data.qvel[vs:vs + BODY_QVEL_SIZE].copy()
        forces = self.data.actuator_force[acts:acts + NUM_ACTUATORS].copy()

        pos_x, pos_y, pos_z, qx, qy, qz, qw = extract_root_pose(qpos)
        joints = extract_joint_states(qpos, qvel, forces)

        # Compute instantaneous metrics
        avg_reward = robot["total_reward"] / max(1, robot["episode_count"])
        avg_episode_len = robot["total_steps"] / max(1, robot["episode_count"])
        uptime_pct = 1.0 - (robot["fall_count"] / max(1, robot["total_steps"] / 50))  # falls per ~1s window

        return {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,
            "quat_x": qx,
            "quat_y": qy,
            "quat_z": qz,
            "quat_w": qw,
            "joints": joints,
            "battery": robot["battery"],
            "status": robot["status"],
            "height": pos_z,
            "steps": self.steps,
            # Performance metrics for platform feedback
            "metrics": {
                "reward": round(robot["reward_accum"], 2),
                "avg_episode_reward": round(avg_reward, 2),
                "avg_episode_length": round(avg_episode_len, 1),
                "episode_steps": robot["episode_steps"],
                "fall_count": robot["fall_count"],
                "episode_count": robot["episode_count"],
                "uptime_pct": round(max(0.0, min(1.0, uptime_pct)), 3),
                "forward_velocity": round((self.data.qpos[robot["qpos_start"]] - robot["prev_x"]), 4),
            },
        }

    def get_object_states(self) -> list[dict]:
        """Get positions and velocities of dynamic objects (red_box, blue_cylinder)."""
        objects = []
        for name in ("red_box", "blue_cylinder"):
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    continue
                pos = self.data.xpos[bid].copy()
                # Velocity: use body com velocity if available
                vel = self.data.cvel[bid][:3].copy() if bid < len(self.data.cvel) else np.zeros(3)
                objects.append({
                    "name": name,
                    "pos_x": float(pos[0]), "pos_y": float(pos[1]), "pos_z": float(pos[2]),
                    "vel_x": float(vel[0]), "vel_y": float(vel[1]), "vel_z": float(vel[2]),
                })
            except Exception:
                pass
        return objects

    def get_all_robot_ids(self) -> list[str]:
        return [r["id"] for r in self.robots]

    def _find_robot(self, robot_id: str) -> dict | None:
        for r in self.robots:
            if r["id"] == robot_id:
                return r
        return None

    def close(self):
        pass
