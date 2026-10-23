"""
MuJoCo lab room robot simulator for FleetOS.

No hardcoded gaits: actions come from trained policies. For locomotion tasks a
robot runs its platform-assigned policy locally every control tick (edge
execution, see edge_policy.py); other tasks use the cloud inference service.
The platform decides which model each robot runs.

Architecture:
  TaskState  — receives gRPC commands, tracks what robot should do
  EdgePolicyManager — runs each robot's assigned policy locally (locomotion)
  InferenceClient — async HTTP client to cloud inference (everything else)
  LabSimulation — MuJoCo physics engine
  TelemetryClient — streams state to platform

Environment variables:
    ROBOT_ID        Initial robot identifier (default: robot-0001)
    GRPC_TARGET     Ingestion gRPC endpoint (default: ingestion:50051)
    INFERENCE_URL   Inference service URL (default: http://inference:8081)
    SIM_STEP_HZ     Physics step rate (default: 50)
    TELEMETRY_HZ    Telemetry send rate (default: 10)
    CONTROL_PORT    HTTP control server port (default: 8085)
    EDGE_POLICY, EDGE_POLICY_TASKS, PLATFORM_API_URL, ... — see edge_policy.py
"""

import logging
import os
import queue
import signal
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    robot_id = os.environ.get("ROBOT_ID", "robot-0001")
    grpc_target = os.environ.get("GRPC_TARGET", "ingestion:50051")
    inference_url = os.environ.get("INFERENCE_URL", "http://inference:8081")
    sim_step_hz = int(os.environ.get("SIM_STEP_HZ", "50"))
    telemetry_hz = int(os.environ.get("TELEMETRY_HZ", "10"))
    control_port = int(os.environ.get("CONTROL_PORT", "8085"))

    log.info("Starting MuJoCo lab simulator robot_id=%s target=%s", robot_id, grpc_target)
    log.info("Physics: %d Hz, Telemetry: %d Hz, Inference: %s", sim_step_hz, telemetry_hz, inference_url)

    from mujoco_env import LabSimulation
    from task_state import TaskState
    from inference_client import InferenceClient
    from telemetry_client import TelemetryClient
    from observation import build_observation_from_mujoco
    from frame_server import start_server, set_spawn_callback, set_list_robots_callback
    from exploration import Explorer
    from transitions import TransitionRecorder
    from edge_policy import EdgePolicyManager, DEFAULT_TASKS, choose_action_source, parse_tasks

    sim = LabSimulation()
    edge_enabled = os.environ.get("EDGE_POLICY", "true").lower() == "true"
    edge_tasks = parse_tasks(os.environ.get("EDGE_POLICY_TASKS", DEFAULT_TASKS))
    edge = EdgePolicyManager.from_env() if edge_enabled else None
    if edge:
        edge.register_robot(robot_id)
        edge.start()
    log.info("Edge policy execution: %s (tasks=%s)", edge_enabled, sorted(edge_tasks))
    action_source: dict[str, str] = {}
    explorer = Explorer.from_env()
    recorders: dict[str, TransitionRecorder] = {}
    log.info("Exploration: fraction=%.2f std=%.2f", explorer.fraction, explorer.std)
    spawn_queue: queue.Queue[str] = queue.Queue()
    spawn_results: dict[str, dict] = {}

    # Per-robot task state (receives commands via gRPC) + inference client
    def _make_task(rid: str) -> TaskState:
        t = TaskState(grpc_target, rid)
        t.start()
        return t

    def _make_inference(rid: str) -> InferenceClient:
        return InferenceClient(inference_url)

    tasks: dict[str, TaskState] = {robot_id: _make_task(robot_id)}
    infer: dict[str, InferenceClient] = {robot_id: _make_inference(robot_id)}
    telemetry = TelemetryClient(grpc_target, robot_id)

    # Spawn handler (HTTP thread → main thread via queue)
    def request_spawn() -> dict:
        req_id = str(time.time())
        spawn_queue.put(req_id)
        for _ in range(20):
            time.sleep(0.1)
            if req_id in spawn_results:
                return spawn_results.pop(req_id)
        return {"error": "spawn timeout"}

    set_spawn_callback(request_spawn)
    set_list_robots_callback(sim.get_all_robot_ids)
    start_server(control_port)

    # Connect telemetry gRPC
    connected = False
    for attempt in range(30):
        try:
            telemetry.connect()
            connected = True
            break
        except Exception as e:
            log.warning("gRPC connect attempt %d failed: %s", attempt + 1, e)
            time.sleep(2)

    if not connected:
        log.error("Failed to connect to ingestion after 30 attempts")
        sys.exit(1)

    # Timing
    step_interval = 1.0 / sim_step_hz
    telemetry_every = max(1, sim_step_hz // telemetry_hz)
    lidar_every = sim_step_hz  # 1 Hz

    running = True

    def shutdown(signum, frame):
        nonlocal running
        log.info("Shutting down (signal %d)", signum)
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    log.info("Simulation loop started")
    step = 0

    try:
        while running:
            loop_start = time.monotonic()

            # Process spawn requests
            while not spawn_queue.empty():
                req_id = spawn_queue.get_nowait()
                result = sim.spawn_robot()
                spawn_results[req_id] = result
                new_rid = result.get("robot_id")
                if new_rid and new_rid not in tasks:
                    tasks[new_rid] = _make_task(new_rid)
                    infer[new_rid] = _make_inference(new_rid)
                if new_rid and edge:
                    edge.register_robot(new_rid)

            # Build observations, request inference, get actions
            robot_ids = sim.get_all_robot_ids()
            actions = {}
            task_commands = {}
            # Telemetry/record ticks: a transition spans one telemetry interval.
            record_tick = step % telemetry_every == 0
            # Completed transitions (offline RL records) to attach to telemetry.
            records: dict[str, list[dict]] = {}
            episodes_before: dict[str, int] = {}

            for rid in robot_ids:
                if rid not in tasks:
                    tasks[rid] = _make_task(rid)
                    infer[rid] = _make_inference(rid)
                    if edge:
                        edge.register_robot(rid)

                task = tasks[rid]
                robot = sim._find_robot(rid)
                if not robot:
                    continue

                # Build unified 86-dim observation
                obs = build_observation_from_mujoco(
                    sim.model, sim.data,
                    robot["qpos_start"], robot["dof_start"],
                    task_command=task.current_task,
                )

                # Locomotion: the robot's assigned policy runs locally on this
                # tick's observation (as in training/evaluation). Otherwise:
                # cloud inference (async, ~10 Hz, latest result held).
                current = task.current_task
                edge_action = edge.action(rid, obs) if edge and current in edge_tasks else None
                source = choose_action_source(current, edge_enabled, edge_action, edge_tasks)
                if action_source.get(rid) != source:
                    log.info("robot %s action source: %s (task=%s)", rid, source, current)
                    action_source[rid] = source
                if source == "edge":
                    policy_action = edge_action
                else:
                    infer[rid].request_action(obs.tolist(), current, goal=task.current_goal)
                    policy_action = infer[rid].get_cached_action()

                # Exploration noise (held per transition) for exploration
                # robots. The applied action is what gets recorded.
                applied, explored = explorer.apply(rid, policy_action, refresh=record_tick)
                actions[rid] = applied
                task_commands[rid] = task.current_task

                rec = recorders.setdefault(rid, TransitionRecorder(rid))
                episodes_before[rid] = robot["episode_count"]
                records[rid] = rec.before_step(
                    obs, applied, explored, robot["cum_reward"], record=record_tick)

            # Step physics with task-conditional reward (falls reset robots inside)
            sim.step(actions, task_commands=task_commands)
            step += 1

            object_states = None
            for rid in robot_ids:
                robot = sim._find_robot(rid)
                if not robot or rid not in recorders:
                    continue
                fell = robot["episode_count"] != episodes_before.get(rid, robot["episode_count"])
                terminal = recorders[rid].after_step(robot["cum_reward"], fell=fell)
                # Telemetry goes out on record ticks, and immediately on a fall
                # so the terminal transition is not lost.
                if not (record_tick or terminal):
                    continue
                state = sim.get_robot_state(rid)
                if not state:
                    continue
                if object_states is None:
                    object_states = sim.get_object_states()
                pending = records.get(rid, []) + terminal
                if not pending:
                    telemetry.send_state(state, robot_id=rid)
                for r in pending:
                    telemetry.send_state(state, robot_id=rid, training_ctx={
                        **r,
                        "task_command": task_commands.get(rid, "idle"),
                        "objects": object_states,
                    })

            # LiDAR at 1 Hz
            if step % lidar_every == 0:
                for rid in robot_ids:
                    telemetry.send_lidar(robot_id=rid)

            # Log progress
            if step % (sim_step_hz * 10) == 0:
                state = sim.get_robot_state(robot_ids[0]) if robot_ids else None
                if state:
                    first_task = tasks[robot_ids[0]].current_task if robot_ids else "idle"
                    log.info("step=%d robots=%d pos=(%.2f, %.2f, %.2f) bat=%.0f%% task=%s",
                             step, len(robot_ids),
                             state["pos_x"], state["pos_y"], state["pos_z"],
                             state["battery"] * 100, first_task)

            # Maintain step rate
            elapsed = time.monotonic() - loop_start
            sleep_time = step_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        log.info("Cleaning up after %d steps", step)
        if edge:
            edge.stop()
        sim.close()
        for t in tasks.values():
            t.close()
        telemetry.close()


if __name__ == "__main__":
    main()
