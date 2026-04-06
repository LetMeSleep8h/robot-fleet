"""
gRPC command receiver and command-to-MuJoCo-action translator.

Opens a persistent StreamCommands gRPC stream to the ingestion service.
Commands arrive as protobuf RobotCommand messages with command_type field.

Translates commands into 25-dim MuJoCo torque arrays (17 body + 4 ankle + 4 gripper).
"""

import logging
import math
import os
import sys
import threading
import time
from typing import Optional

import grpc
import numpy as np

# Generated protobuf imports (generated at Docker build time into gen/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gen"))
from proto.telemetry_pb2 import CommandRequest
from proto.telemetry_pb2_grpc import TelemetryServiceStub

from joint_mapping import FLEET_TO_MUJOCO_ACTUATOR

log = logging.getLogger(__name__)

NUM_ACTUATORS = 25

# Actuator indices (matching lab_room.xml actuator order)
# Body: 0-16 (original 17)
ABDOMEN_Y, ABDOMEN_Z, ABDOMEN_X = 0, 1, 2
R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE = 3, 4, 5, 6
L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE = 7, 8, 9, 10
R_SHOULDER1, R_SHOULDER2, R_ELBOW = 11, 12, 13
L_SHOULDER1, L_SHOULDER2, L_ELBOW = 14, 15, 16
# Ankles: 17-20
R_ANKLE_Y, R_ANKLE_X = 17, 18
L_ANKLE_Y, L_ANKLE_X = 19, 20
# Grippers: 21-24
R_GRIP_L, R_GRIP_R = 21, 22
L_GRIP_L, L_GRIP_R = 23, 24


class CommandHandler:
    """Receives commands via gRPC StreamCommands and produces MuJoCo actions."""

    def __init__(self, grpc_target: str, robot_id: str):
        self.robot_id = robot_id
        self.current_command = "idle"
        self.command_params: dict = {}
        self.command_tick = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._grpc_target = grpc_target
        self._channel = grpc.insecure_channel(grpc_target)
        self._stub = TelemetryServiceStub(self._channel)
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the gRPC command stream in a background thread."""
        self._thread = threading.Thread(target=self._subscribe, daemon=True)
        self._thread.start()
        log.info("Command handler started for %s", self.robot_id)

    def _subscribe(self):
        """Maintain a persistent StreamCommands gRPC stream with auto-reconnect."""
        while not self._stop_event.is_set():
            try:
                request = CommandRequest(robot_id=self.robot_id)
                stream = self._stub.StreamCommands(request)
                log.info("StreamCommands connected for %s", self.robot_id)

                for robot_cmd in stream:
                    if self._stop_event.is_set():
                        return
                    self._handle_command(robot_cmd)

                # Stream ended normally (server closed)
                log.info("Command stream closed for %s", self.robot_id)
            except grpc.RpcError as e:
                if self._stop_event.is_set():
                    return
                if e.code() == grpc.StatusCode.CANCELLED:
                    return
                log.warning("Command stream error for %s: %s — reconnecting in 2s",
                            self.robot_id, e.code())
                time.sleep(2)

    def _handle_command(self, robot_cmd):
        """Translate a protobuf RobotCommand into internal command state."""
        cmd_type = robot_cmd.command_type  # "move", "wave", "dance", etc.
        params = {}

        action_field = robot_cmd.WhichOneof("action")
        if action_field == "move":
            move = robot_cmd.move
            params = {
                "target_position": {
                    "x": move.target_position.x,
                    "y": move.target_position.y,
                    "z": move.target_position.z,
                } if move.target_position else {},
                "max_velocity": move.max_velocity,
            }
            if not cmd_type:
                cmd_type = "move"
        elif action_field == "stop":
            params = {"emergency": robot_cmd.stop.emergency}
            if not cmd_type:
                cmd_type = "stop"
        elif action_field == "joint":
            # Gesture commands (wave, dance, bow, etc.) arrive as JointCommand
            # with command_type set to the gesture name
            joint = robot_cmd.joint
            params = {"duration": joint.duration_seconds}
            # If joint targets are provided, convert to apply_actions format
            if joint.target_joints:
                params["actions"] = [
                    {"joint": js.name, "torque": js.torque}
                    for js in joint.target_joints
                ]
                if not cmd_type:
                    cmd_type = "apply_actions"
            elif not cmd_type:
                cmd_type = "idle"  # JointCommand with no type and no targets

        if not cmd_type:
            cmd_type = "idle"

        with self._lock:
            self.current_command = cmd_type
            self.command_params = params
            self.command_tick = 0

        log.info("Command received via gRPC: %s params=%s", cmd_type, params)

    def get_action(self, sim_time: float) -> np.ndarray:
        """Produce a 25-dim action array based on current command.

        Args:
            sim_time: Current simulation time in seconds.

        Returns:
            np.ndarray of shape (25,) with values in [-1, 1].
        """
        with self._lock:
            cmd = self.current_command
            params = self.command_params
            self.command_tick += 1
            tick = self.command_tick

        if cmd == "stop":
            return np.zeros(NUM_ACTUATORS, dtype=np.float32)

        if cmd == "apply_actions":
            return self._apply_inference_actions(params)

        if cmd in ("move", "move_relative", "walk"):
            return self._walk_action(sim_time)

        if cmd == "dance":
            return self._dance_action(sim_time)

        if cmd == "wave":
            return self._wave_action(sim_time)

        if cmd == "jump":
            return self._jump_action(sim_time)

        if cmd == "bow":
            return self._bow_action(sim_time)

        if cmd in ("sit", "crouch"):
            return self._sit_action(sim_time)

        if cmd == "look_around":
            return self._look_action(sim_time)

        if cmd == "stretch":
            return self._stretch_action(sim_time)

        # idle: small stabilization torques
        return self._idle_action(sim_time)

    def _apply_inference_actions(self, params: dict) -> np.ndarray:
        """Apply joint torques from inference predicted_actions directly."""
        action = self._idle_action(0)  # start from stable base
        actions = params.get("actions", [])
        for a in actions:
            joint_name = a.get("joint", "")
            torque = a.get("torque", 0.0)
            if joint_name in FLEET_TO_MUJOCO_ACTUATOR:
                idx = FLEET_TO_MUJOCO_ACTUATOR[joint_name]
                action[idx] = np.clip(torque, -1.0, 1.0)
        return action

    def _idle_action(self, t: float) -> np.ndarray:
        """Standing pose: target joint angles in radians.

        With position actuators, ctrl = desired joint angle.
        0.0 = neutral/upright for all joints.
        """
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        # All zeros = upright standing pose (PD controller holds it)
        return action

    def _walk_action(self, t: float) -> np.ndarray:
        """Cyclic joint angle targets for forward walking gait.

        With position actuators, ctrl = target angle (radians).
        Uses small offsets from the standing pose (0).
        """
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        phase = t * 1.2 * 2 * math.pi

        # Hip pitch: alternating leg swing (±0.4 rad ≈ ±23 deg)
        action[R_HIP_Y] = -0.4 * math.sin(phase)
        action[L_HIP_Y] = 0.4 * math.sin(phase)

        # Knee: flex during swing (negative range in XML)
        action[R_KNEE] = -0.5 * max(0, math.sin(phase))
        action[L_KNEE] = -0.5 * max(0, -math.sin(phase))

        # Hip roll: lateral weight shift
        action[R_HIP_X] = 0.05 * math.sin(phase * 0.5)
        action[L_HIP_X] = -0.05 * math.sin(phase * 0.5)

        # Ankle pitch: push-off during stance
        action[R_ANKLE_Y] = 0.2 * math.sin(phase)
        action[L_ANKLE_Y] = -0.2 * math.sin(phase)

        # Arm counter-swing
        action[R_SHOULDER1] = 0.3 * math.sin(phase)
        action[L_SHOULDER1] = -0.3 * math.sin(phase)

        # Slight forward lean
        action[ABDOMEN_Y] = -0.1

        return action

    def _dance_action(self, t: float) -> np.ndarray:
        """Rhythmic full-body movement."""
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        phase = t * 2.5 * 2 * math.pi

        action[ABDOMEN_Z] = 0.4 * math.sin(phase)
        action[ABDOMEN_Y] = 0.2 * math.sin(phase * 2)
        action[R_SHOULDER1] = 0.5 * math.sin(phase)
        action[L_SHOULDER1] = 0.5 * math.sin(phase + 1)
        action[R_ELBOW] = 0.3 * math.sin(phase * 2)
        action[L_ELBOW] = 0.3 * math.sin(phase * 2 + 1)
        action[R_HIP_Y] = 0.3 * math.sin(phase)
        action[L_HIP_Y] = 0.3 * math.sin(phase)
        action[R_KNEE] = 0.2 * abs(math.sin(phase))
        action[L_KNEE] = 0.2 * abs(math.sin(phase))
        # Ankle support during dance
        action[R_ANKLE_Y] = 0.15 * math.sin(phase)
        action[L_ANKLE_Y] = 0.15 * math.sin(phase)
        return action

    def _wave_action(self, t: float) -> np.ndarray:
        """Wave the right arm."""
        action = self._idle_action(t)
        phase = t * 3 * 2 * math.pi
        action[R_SHOULDER1] = -0.7
        action[R_SHOULDER2] = 0.3
        action[R_ELBOW] = 0.5 * math.sin(phase)
        return action

    def _jump_action(self, t: float) -> np.ndarray:
        """Attempt a jump by extending legs forcefully."""
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        tick = self.command_tick

        if tick < 25:  # Crouch
            action[R_HIP_Y] = -0.8
            action[L_HIP_Y] = -0.8
            action[R_KNEE] = 0.8
            action[L_KNEE] = 0.8
            action[R_ANKLE_Y] = -0.4
            action[L_ANKLE_Y] = -0.4
        elif tick < 35:  # Explode upward
            action[R_HIP_Y] = 0.9
            action[L_HIP_Y] = 0.9
            action[R_KNEE] = -0.9
            action[L_KNEE] = -0.9
            action[R_ANKLE_Y] = 0.9  # Push off ground
            action[L_ANKLE_Y] = 0.9
        else:  # Recover
            action[ABDOMEN_Y] = 0.1
            action[R_ANKLE_Y] = 0.1
            action[L_ANKLE_Y] = 0.1
            if tick > 75:
                with self._lock:
                    self.current_command = "idle"
        return action

    def _bow_action(self, t: float) -> np.ndarray:
        """Bow forward from the waist."""
        action = self._idle_action(t)
        tick = self.command_tick
        bow_amount = min(1.0, tick / 50) if tick < 50 else max(0.0, 1.0 - (tick - 75) / 50)
        action[ABDOMEN_Y] = 0.5 * bow_amount
        action[R_HIP_Y] = -0.2 * bow_amount
        action[L_HIP_Y] = -0.2 * bow_amount
        action[R_ANKLE_Y] = 0.2 * bow_amount  # Counterbalance
        action[L_ANKLE_Y] = 0.2 * bow_amount
        if tick > 125:
            with self._lock:
                self.current_command = "idle"
        return action

    def _sit_action(self, t: float) -> np.ndarray:
        """Bend knees to sit."""
        action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        sit = min(1.0, self.command_tick / 50)
        action[R_HIP_Y] = -0.6 * sit
        action[L_HIP_Y] = -0.6 * sit
        action[R_KNEE] = 0.7 * sit
        action[L_KNEE] = 0.7 * sit
        action[ABDOMEN_Y] = 0.15 * sit
        action[R_ANKLE_Y] = 0.3 * sit  # Keep feet flat
        action[L_ANKLE_Y] = 0.3 * sit
        return action

    def _look_action(self, t: float) -> np.ndarray:
        """Pan head left and right."""
        action = self._idle_action(t)
        action[ABDOMEN_Z] = 0.4 * math.sin(t * 0.8 * 2 * math.pi)
        action[ABDOMEN_Y] = 0.1 * math.sin(t * 0.4 * 2 * math.pi)
        if self.command_tick > 200:
            with self._lock:
                self.current_command = "idle"
        return action

    def _stretch_action(self, t: float) -> np.ndarray:
        """Arms up stretch."""
        action = self._idle_action(t)
        stretch = min(1.0, self.command_tick / 50)
        action[R_SHOULDER1] = -0.7 * stretch
        action[L_SHOULDER1] = -0.7 * stretch
        action[R_ELBOW] = -0.3 * stretch
        action[L_ELBOW] = -0.3 * stretch
        action[ABDOMEN_Z] = 0.2 * math.sin(t * 0.5 * 2 * math.pi) * stretch
        if self.command_tick > 200:
            with self._lock:
                self.current_command = "idle"
        return action

    def close(self):
        self._stop_event.set()
        if self._channel:
            self._channel.close()
