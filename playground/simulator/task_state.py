"""
Task state tracker — receives gRPC commands and tracks what the robot
should be doing. Does NOT produce actions (inference does that).

Replaces the old command_handler.py which had hardcoded gaits.
"""

import logging
import os
import sys
import threading
import time
from typing import Optional

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gen"))
from proto.telemetry_pb2 import CommandRequest
from proto.telemetry_pb2_grpc import TelemetryServiceStub

log = logging.getLogger(__name__)


class TaskState:
    """Tracks the current task for a robot. Updated via gRPC command stream."""

    def __init__(self, grpc_target: str, robot_id: str):
        self.robot_id = robot_id
        self.task_type = "idle"
        self.goal: Optional[list] = None
        self.command_params: dict = {}
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
        log.info("Task state tracker started for %s", self.robot_id)

    def _subscribe(self):
        """Persistent gRPC StreamCommands with auto-reconnect."""
        while not self._stop_event.is_set():
            try:
                request = CommandRequest(robot_id=self.robot_id)
                stream = self._stub.StreamCommands(request)
                log.info("StreamCommands connected for %s", self.robot_id)

                for robot_cmd in stream:
                    if self._stop_event.is_set():
                        return
                    self._handle_command(robot_cmd)

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
        """Extract task type and goal from a gRPC command."""
        cmd_type = robot_cmd.command_type or "idle"
        goal = None
        params = {}

        action_field = robot_cmd.WhichOneof("action")
        if action_field == "move":
            move = robot_cmd.move
            if move.target_position:
                goal = [move.target_position.x, move.target_position.y, move.target_position.z]
            params = {"max_velocity": move.max_velocity}
            if not cmd_type or cmd_type == "idle":
                cmd_type = "move"

        elif action_field == "stop":
            cmd_type = "idle"
            params = {"emergency": robot_cmd.stop.emergency}

        elif action_field == "joint":
            joint = robot_cmd.joint
            params = {"duration": joint.duration_seconds}
            if joint.target_joints:
                params["target_joints"] = [
                    {"name": js.name, "position": js.position, "torque": js.torque}
                    for js in joint.target_joints
                ]

        with self._lock:
            self.task_type = cmd_type
            self.goal = goal
            self.command_params = params

        log.info("Task updated: %s → %s goal=%s", self.robot_id, cmd_type, goal)

    @property
    def current_task(self) -> str:
        with self._lock:
            return self.task_type

    @property
    def current_goal(self) -> Optional[list]:
        with self._lock:
            return self.goal

    def close(self):
        self._stop_event.set()
        if self._channel:
            self._channel.close()
