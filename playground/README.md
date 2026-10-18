# FleetOS Playground

Robot simulation environment for the FleetOS platform. The playground acts like a **real robot** — it runs physics, sends telemetry (including performance metrics), and executes commands. All reasoning (inference, model management, billing) happens in the platform.

FleetOS uses a **cloud-brain architecture**: high-level reasoning in the cloud, execution on the robot.

## Architecture

```
Playground (Robot)                      Platform (Cloud Brain)
──────────────────                      ──────────────────────
Simulator (MuJoCo Humanoid-v4)
  ├── gRPC StreamTelemetry ──────────►  Ingestion :50051
  │   (20-joint state + PerformanceMetrics)  ├── Kafka (robot.telemetry)
  │   reward, uptime, falls, velocity        └── Processor → Postgres/Redis/S3
  │                                                └── Aggregates success_rate → model registry
  ├── gRPC StreamCommands  ◄──────────  Ingestion :50051
  │   (move, wave, dance, inference)        └── Kafka (robot.commands.dispatch)
  └── HTTP :8085 (/spawn, /robots)              → CommandDispatcher → gRPC
                                                → Kafka (robot.command-acks) → Temporal
Web UI :5173
  ├── Three.js 3D simulation view       API :8080
  ├── AI Inference Pipeline panel         ├── /api/v1/inference (per-robot model lookup)
  ├── Command Feed (live polling)         ├── Admin Console at /admin/
  └── Model Performance metrics           └── Temporal :7233 (workflows)

Inference :8081 (SB3 PPO, runs in platform stack)
  ├── Serves robot's assigned model (inference_model_id)
  ├── Hot-swaps from S3 on model registry deploy
  └── Polls platform API for new deployments
```

## What the Playground Does NOT Do

- No inference (runs in platform)
- No model management (platform model registry + canary deployment)
- No direct Redis or Kafka access (commands arrive via gRPC `StreamCommands`)
- No billing (platform Temporal workflows)
- No training (handled by `platform/training/`)

## Quick Start

### With Platform on Docker Compose

```bash
cd ../platform && docker compose up -d
docker compose up -d
open http://localhost:5173
```

### With Platform on Kubernetes (ArgoCD)

When the platform runs on a kind cluster, the playground connects via NodePorts:

```bash
# Start platform on k8s
cd ../platform && make k8s-up

# Start playground (connects to platform NodePorts automatically)
docker compose up -d
open http://localhost:5173
```

The playground `docker-compose.yml` uses `host.docker.internal` to reach the platform k8s NodePorts (ingestion :30051, inference :30081, API :30880).

### With Platform on k3s (Colima)

k3s runs inside the Colima VM, so NodePorts are reached at the VM's IP (`colima list`), not `host.docker.internal`. The `docker-compose.k3s.yml` override sets that up:

```bash
PLATFORM_HOST=<colima-vm-ip> docker-compose -f docker-compose.yml -f docker-compose.k3s.yml up -d simulator web
open http://localhost:5173
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `simulator` | 8085 | MuJoCo physics (7x sub-stepping for real-time), gRPC telemetry + commands |
| `web` | 5173 | React dashboard (Three.js 3D, inference pipeline, command feed, performance metrics) |
| `ros2-bridge` | — | Bridges platform telemetry to ROS 2 topics |

## Performance Metrics

The simulator computes **Humanoid-v4 reward** every physics step and streams it back to the platform via protobuf `PerformanceMetrics`:

| Metric | Formula | Purpose |
|--------|---------|---------|
| `reward` | forward_velocity + alive_bonus(5.0) - control_cost(0.1 * ctrl^2) | Current episode accumulated reward |
| `uptime_pct` | time_standing / total_time | Becomes `success_rate` in model registry for canary evaluation |
| `fall_count` | total resets (height < 0.4m) | Stability indicator |
| `forward_velocity` | delta_x / dt | Locomotion quality |

The platform processor aggregates `uptime_pct` across all robots per model every 30s and updates the model registry's `success_rate`. This is what `CompareCanaryMetrics` uses during canary deployments — if the new model's success_rate degrades >10% vs baseline, it auto-rolls back.

## Command Flow

```
Web UI ("walk forward") → POST /api/v1/inference
  → API looks up robot's inference_model_id → sends to inference:8081
  → SB3 PPO policy predicts 17-dim action vector
  → Known command ("walk") resolved directly (smoother motion)
  → API publishes to Kafka robot.commands
  → Processor starts Temporal CommandDispatchWorkflow
  → Workflow: WriteAudit → PublishCommand → Kafka robot.commands.dispatch
  → Ingestion CommandDispatcher routes to robot's gRPC stream
  → Simulator CommandHandler receives "walk"
  → _walk_action(sim_time) produces cyclic gait torques
  → MuJoCo physics (7 sub-steps × 0.003s = real-time)
  → Robot walks, telemetry streams back with reward metrics
```

## Web Dashboard Features

- **3D Simulation View** — Three.js rendering of MuJoCo humanoid in lab room
- **AI Inference Pipeline** — 4-step visualization: instruction → inference → dispatch → robot ack (with timing)
- **Command Feed** — Live-polling command history with status badges and flash animation on new commands
- **Model Performance** — Real-time reward, avg episode reward, uptime %, velocity, falls, episode count
- **Joint Visualizer** — Per-joint angle bars with torque indicators
- **Telemetry Stream** — Raw telemetry event log

## Fleet Experience for Offline Training

Every robot's telemetry carries a training context, which the platform turns into the dataset its training pipeline learns from (no simulator in training):

```
Simulator (50 Hz control loop, 10 Hz telemetry)
  │  per telemetry record (training context):
  │    observation = exact 86-dim policy input at this step
  │    action      = 25-dim action actually applied (after exploration noise, clipped)
  │    reward      = reward earned by that action until the robot's next record
  │    episode_id / step, episode_done (a terminal record is sent on every fall)
  │    exploration = whether the action included noise
  │
  gRPC telemetry → Ingestion → Kafka → Processor
  │
  S3: fleetos-telemetry/training_data/v2/{YYYY/MM/DD/HH}/{robot_id}/transitions_{ns}.ndjson
```

A deterministic fraction of robots (`EXPLORATION_FRACTION`, default 0.2) adds Gaussian noise (`EXPLORATION_STD`, default 0.1) to their actions so the data isn't limited to what the current policy already does. The platform pipeline (`POST /api/v1/training/pipelines`) snapshots these batches and trains offline (CQL); MuJoCo is only used to evaluate the result.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_ID` | robot-0001 | Initial robot identifier |
| `GRPC_TARGET` | ingestion:50051 | Ingestion gRPC endpoint |
| `SIM_STEP_HZ` | 50 | Control loop rate (physics sub-steps 7x for real-time) |
| `TELEMETRY_HZ` | 10 | Telemetry send rate |
| `CONTROL_PORT` | 8085 | HTTP control server port |
| `EXPLORATION_FRACTION` | 0.2 | Share of robots (chosen deterministically by ID) that add exploration noise |
| `EXPLORATION_STD` | 0.1 | Std of the per-dimension Gaussian exploration noise |
| `INFERENCE_URL` | http://inference:8081 | Inference service URL (use http://host.docker.internal:30081 for k8s) |
