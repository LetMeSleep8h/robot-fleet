# FleetOS

A distributed robot fleet management platform built on a **cloud-brain architecture**: high-level reasoning happens in the platform (cloud); robots handle execution only. Robots connect via gRPC to send telemetry and receive commands. The platform runs inference, manages model lifecycle (staged → canary → deployed), and provides developer-facing REST/WebSocket APIs with auth, billing, and full-stack analytics.

## Features

- **Real-time Telemetry Pipeline** -- Humanoid robots stream 20-DOF joint states, LiDAR, video, and audio over gRPC into Kafka, processed into Redis (hot state) + S3 (raw storage)
- **Developer API Platform** -- REST, WebSocket, and gRPC APIs with OpenAPI 3.1 spec, TypeScript + Python SDKs
- **Multi-tenant Auth & Billing** -- JWT + OAuth2/OIDC + API key authentication, RBAC (4 roles), per-tenant rate limiting, usage metering, 3-tier pricing (free/pro/enterprise)
- **Edge policy execution** -- robots run their assigned walking policy locally every control tick (the timing it was trained at); the platform decides which model via `GET /api/v1/robots/{id}/policy` (canary assignment, else the deployed fleet model)
- **AI Inference** -- SB3 PPO/SAC policy serving for API requests and non-locomotion commands; applies each model's observation normalisation; instruction bias only for the zero-action stub; loads models from S3/MinIO, hot-swaps on deployment
- **Durable Command Pipeline** -- Commands published to Kafka (`robot.commands`), consumed by processor, orchestrated via Temporal workflows with audit trail, ack tracking, retries, and timeout handling
- **Semantic Commands** -- Natural language robot control via extensible command registry (strategy pattern), with FAISS-backed resolution for unmatched instructions (searches fleet state to resolve spatial commands like "go where the idle robots are")
- **Model Registry** -- Model lifecycle management (staged -> canary -> deployed -> archived) with S3 artifact storage, canary deployment via Temporal workflows
- **Admin Console** -- React SPA at `/admin/` for tenant management, DB-backed API key lifecycle (create/revoke with SHA-256 hashing), billing overview
- **Temporal Billing** -- Per-tenant BillingCycleWorkflow: 6h usage aggregation (Redis → Postgres), monthly invoice generation, payment processing with dunning retries, tier change signals
- **Evaluation Gate + Canary Deployment** -- Deploys are refused unless the model's eval pass rate ≥ 0.5 and its reward is within 5% of the deployed model; accepted deploys roll out progressively (assign 5% → observe → 25% → 50% → 100%) with deterministic robot selection (fnv32 hash), live success_rate comparison from robot uptime metrics, automatic rollback
- **Per-Robot Model Assignment** -- Each robot tracks its `inference_model_id`; API inference requests are pinned to that model's artifact (inference keeps an LRU cache of policies), so canary robots are served by the canary policy. The simulator calls inference directly and always gets the active model
- **Performance Metrics Feedback** -- Simulator computes Humanoid-v4 reward (forward velocity + alive bonus - control cost), tracks falls/uptime, streams metrics via protobuf → processor aggregates into model registry `success_rate`
- **Kafka Event Pipeline** -- Commands, acks, and deployment events all flow through Kafka topics (no Redis pub/sub for events); Redis is cache + rate limiter only
- **Temporal Workflow Orchestration** -- Command dispatch, model deployment, agent deployment, billing cycles, and webhook delivery all run as durable Temporal workflows with full visibility via Temporal UI
- **FAISS Vector Search** -- Semantic search over robot fleet state via `/resolve` endpoint, indexes robots from Redis hot state every 5s, resolves natural language instructions to concrete commands with spatial context
- **Analytics Pipeline** -- S3 (raw NDJSON) -> Spark (batch) -> ClickHouse (OLAP) -> Redis (cache) -> API
- **ROS 2 Integration** -- Bridge node publishing to standard ROS 2 topics (JointState, PoseStamped, BatteryState, LaserScan)
- **Cloud-native Infrastructure** -- Kubernetes, Helm, Terraform (AWS), Docker Compose, Istio service mesh (mTLS)
- **Full Observability** -- Prometheus metrics (11 custom), Grafana dashboards (2), alerting rules (8), OpenTelemetry tracing

## Architecture

```
                    Internet
     +-----------------+------------------+
     |                 |                  |
Robot Fleet     CloudFront + WAF      NLB (TCP)
(1000s)          (HTTPS)            (gRPC/TLS)
     |                 |                  |
=====|=================|==================|===============
     |          Istio Ingress GW          |
     |          (ALB + TLS term)          |
     |                 |                  |
     |     +-----------+-----------+      |
     |     |                       |      |
     |  API Service          Ingestion Service
     |  HPA: 3-20 pods      HPA: 2-10 pods
     |  +Envoy sidecar      +Envoy sidecar
     |     |                       |
     |     v                       v
     |  Inference (GPU)      MSK (Kafka 3 brokers)
     |  Ray Serve                  |
     |  +Envoy sidecar            v
     |                       Processor -> S3/MinIO (raw NDJSON)
     |                                 -> Redis (hot state)
     |                                 -> Postgres (metadata)
     |                                        |
     |                              Spark (batch analytics)
     |                                        |
     |                              ClickHouse (OLAP)
     |                                        |
     |                              Redis (cached queries)
     |                                        |
=====|========================================|===============
     |                                        |
Observability                            CI/CD
OTel -> Prom -> Grafana -> PagerDuty     GitHub -> ECR -> ArgoCD -> EKS

AWS Managed: RDS Postgres (Multi-AZ, encrypted), ElastiCache Redis (HA),
             S3 (SSE, lifecycle), EKS (3 node groups: general/GPU/Kafka)
```

## Quick Start

### Prerequisites

- Go 1.26+
- Docker & Docker Compose
- Protocol Buffers compiler (`protoc`)

### Run Locally

```bash
# Option A: Docker Compose
../start.sh                   # Both platform + playground
docker compose up -d          # Platform only

# Option B: Kubernetes (kind + ArgoCD) — fully automated, no port-forwards
make k8s-up                   # Full k8s stack with ArgoCD + Argo Workflows
make k8s-status               # Check pod status
make k8s-logs SVC=api         # Tail logs
make k8s-sync                 # Force ArgoCD re-sync after code changes
make k8s-reload SVC=api       # Rebuild + reload image + restart pod
make k8s-train TASK=locomotion TIMESTEPS=50000  # Submit training workflow
make k8s-argocd-password      # Get ArgoCD admin password
make k8s-down                 # Delete cluster

# K8s Dashboards (all accessible automatically — no port-forward needed)
# ArgoCD:          http://localhost:30080
# Argo Workflows:  https://localhost:30746
# Grafana:         http://localhost:30030  (admin/admin)
# Prometheus:      http://localhost:30090
# API:             http://localhost:30880

# Verify (Docker Compose / Kubernetes)
curl http://localhost:8080/healthz    # Docker Compose
curl http://localhost:30880/healthz   # Kubernetes (NodePort)
# {"status":"ok"}

# List robots
curl -H "X-API-Key: dev-key-001" http://localhost:8080/api/v1/robots

# Send a command
curl -X POST -H "X-API-Key: dev-key-001" \
  -d '{"type":"dance","params":{}}' \
  http://localhost:8080/api/v1/robots/robot-0001/command

# Semantic command (natural language)
curl -X POST -H "X-API-Key: dev-key-001" \
  -d '{"instruction":"wave hello"}' \
  http://localhost:8080/api/v1/robots/robot-0001/semantic-command

# AI inference
curl -X POST -H "X-API-Key: dev-key-001" \
  -d '{"instruction":"pick up the red block"}' \
  http://localhost:8080/api/v1/inference

# Fleet analytics (from ClickHouse, cached in Redis)
curl -H "X-API-Key: dev-key-001" \
  "http://localhost:8080/api/v1/analytics/fleet?from=2026-03-28"

# Run Spark analytics job
docker exec robot-fleet-spark-master-1 /opt/spark-jobs/run.sh

# View dashboards
open http://localhost:5173   # Web playground
open http://localhost:3000   # Grafana (admin/admin)
open http://localhost:8088   # Spark UI
open http://localhost:9001   # MinIO Console (fleetos/fleetos123)
open http://localhost:8123   # ClickHouse (direct SQL)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/admin/` | Admin console (React SPA) |
| `GET` | `/api/v1/robots` | List robots (paginated, includes model assignment) |
| `GET` | `/api/v1/robots/{id}` | Get robot (hot state + inference_model_id) |
| `POST` | `/api/v1/robots/{id}/command` | Send command → Kafka → Temporal workflow |
| `GET` | `/api/v1/robots/{id}/commands` | Command history with audit trail |
| `GET` | `/api/v1/robots/{id}/policy` | Policy the robot should run locally (canary assignment, else deployed fleet model; 404 if none) |
| `POST` | `/api/v1/inference` | AI inference (uses robot's assigned model) |
| `GET` | `/api/v1/fleet/metrics` | Aggregated fleet stats |
| `POST` | `/api/v1/models` | Register model (staged) |
| `POST` | `/api/v1/models/{id}/deploy` | Gate + start canary → `202 {workflow_id}`; `422` gate rejection, `409` not staged, `503` without Temporal |
| `POST` | `/api/v1/training/pipelines` | Start offline training pipeline (`task_type`, `gradient_steps`, `min_transitions`, `max_dataset_files`, `backend: argo\|kubeflow`, `use_katib`, `auto_deploy`) → `202` |
| `GET` | `/api/v1/training/pipelines/{id}` | Pipeline status + result |
| `POST` | `/api/v1/locomotion/jobs` | Submit a single Argo training job (`503` if training not configured) |
| `GET` | `/api/v1/billing/subscription` | Current subscription |
| `PUT` | `/api/v1/billing/subscription/tier` | Change tier (signals workflow) |
| `GET` | `/api/v1/billing/invoices` | Invoice history |
| `POST` | `/api/v1/admin/tenants` | Create tenant + API key (admin only) |
| `GET` | `/api/v1/admin/tenants` | List tenants (admin only) |
| `POST` | `/api/v1/admin/tenants/{id}/keys` | Generate API key (admin only) |
| `DELETE` | `/api/v1/admin/keys/{hash}` | Revoke API key (admin only) |
| `GET` | `/api/v1/analytics/fleet` | Fleet hourly metrics (ClickHouse) |
| `GET` | `/api/v1/ws/telemetry` | WebSocket real-time stream |

### Authentication

Three methods supported:
- **API Key**: `X-API-Key` header (DB-backed with SHA-256 hashing, or hardcoded dev keys)
- **JWT Bearer**: `Authorization: Bearer <token>` (HS256)
- **OAuth2/OIDC**: Bearer token validated against issuer's JWKS

Dev keys: `dev-key-001` (admin/enterprise), `dev-key-002` (viewer/free)

RBAC: `admin`, `operator`, `developer`, `viewer`. Admin routes under `/api/v1/admin/` require `RequireRole(admin)`.

## Data Pipeline

```
Robots -> gRPC stream -> Ingestion -> Kafka (partitioned by robot_id)
                                         |
                                    Processor
                                    /    |    \
                              S3/MinIO  Redis  Postgres
                              (NDJSON)  (hot)  (metadata)
                                 |
                            Spark (batch)
                                 |
                            ClickHouse (OLAP)
                                 |
                            Redis (cache, 60s TTL)
                                 |
                            API endpoints
```

**Why this architecture:**
- Postgres handles metadata only (~2 writes/sec even at 1000 robots)
- S3 handles raw telemetry at any scale (partitioned by time/robot/event)
- ClickHouse handles analytical queries (pre-aggregated by Spark)
- Redis serves hot state + caches OLAP results for sub-ms API latency

## Command Pipeline (Kafka + Temporal)

```
REST API /command ──┐
                    ├──> Kafka (robot.commands) ──> Processor ──> Temporal Workflow
REST API /inference ┘    (keyed by robot_id)                           |
                                                         ┌─────────────┼─────────────┐
                                                         v             v             v
                                                    WriteAudit    [RunInference]  PublishCommand
                                                    (Postgres)     (optional)     (Kafka dispatch topic)
                                                                                      |
                                                                      CommandDispatcher (1 consumer, N robots)
                                                                                      |
                                                                          gRPC StreamCommands -> Robot
                                                                                      |
                                                                      Kafka robot.command-acks
                                                                                      |
                                                                      AckBridge (Kafka consumer) -> Signal Workflow
```

**Kafka topics:**
- `robot.commands` — API → processor (triggers Temporal workflow)
- `robot.commands.dispatch` — Temporal activity → ingestion (commands to robots)
- `robot.command-acks` — ingestion → worker (robot acknowledgments back to Temporal)
- `deployment.events` — Temporal activities → SSE consumers (agent deployment progress)

All event routing goes through Kafka. Redis is used only for cache (hot state, rate limiting, usage counters, dedup).

**Workflow lifecycle:** requested → dispatched → acked/timeout

## AI Inference Pipeline

The inference service runs as a platform service (not on the robot); robots only execute — all reasoning is cloud-side.

```
Instruction ("wave hello") + Optional observation (376-dim MuJoCo state)
    |
SB3 PPO Policy -- predict() → 17-dim MuJoCo action vector
    |
Instruction Bias -- only when no trained model is loaded (zero-action stub);
                    a learned policy's actions are returned unmodified
    |
MuJoCo-to-Fleet Mapping -- 17 actuators → 20 joint schema
    |
Output: 20 joint actions (position, velocity, torque)
    |
API publishes CommandMessage to Kafka (robot.commands)
    |
Processor starts Temporal workflow → Redis → Ingestion → gRPC StreamCommands
```

**Model lifecycle**: Training (`platform/training/`) → S3 upload → Model Registry (staged → canary → deployed) → Inference hot-swaps model from S3.

Watch inference: `docker logs -f platform-inference-1`

## ML Training Pipeline (offline, from fleet experience)

The pipeline learns from what the robots actually did — not from a simulator. It is orchestrated by Temporal and runs on Argo Workflows (default) or Kubeflow Trainer, optionally after a Katib hyperparameter search. MuJoCo is used only as the evaluation "test track". Start a run with `POST /api/v1/training/pipelines` and poll `GET /api/v1/training/pipelines/{id}`.

```
Temporal TrainingPipelineWorkflow (fleetos-training queue)
  │
  ├─ SnapshotDataset — lists s3://fleetos-telemetry/training_data/v2/ (TELEMETRY_BUCKET),
  │     keeps the newest max_dataset_files batches, requires min_transitions (default 1000),
  │     writes datasets/<job>/manifest.json to the models bucket
  │     → too little data stops here (phase "insufficient_data", not a failure)
  │
  ├─ [use_katib] RunHyperparameterSearch ─► Katib Experiment (namespace fleetos-training)
  │     TPE, 4 trials × 2 parallel (HPO_* / kubeflow.hpo values); each trial =
  │     train_offline.py --no-upload on the snapshot (2,000 updates) + 5 test-track episodes
  │     search: learning_rate [1e-4, 1e-3], cql_alpha [0.1, 5.0], batch_size {128,256,512}
  │     objective: eval_mean_reward (StdOut collector)     → best params
  │
  ├─ SubmitTrainingRun (heartbeats; retries resume the same run)
  │     train_offline.py — CQL-SAC on the snapshot, raw observations, no simulator
  │     → evaluate_policy.py on the MuJoCo test track (FleetOS-Humanoid-v2)
  │     backend=argo     ─► Argo WorkflowTemplate fleetos-offline-training
  │     backend=kubeflow ─► Kubeflow Trainer TrainJobs on ClusterTrainingRuntime fleetos-sb3
  │                         (train TrainJob → evaluate TrainJob)
  │     Artifacts → S3: training/<job>/policy.zip, evaluations/<job>/eval_results.json
  │
  ├─ EvaluateTrainedModel — reads this job's eval results; gate (internal/modelgate):
  │     eval pass rate ≥ MODEL_MIN_EVAL_PASS_RATE (0.5) and reward ≤ 5% below deployed model
  │     → rejected models stop here (phase "gated", with the reason)
  │
  ├─ RegisterTrainedModel → registry (staged, model cql-fleet, algorithm cql-sac) with eval
  │     metrics, backend, hyperparameters and data lineage (dataset_manifest, dataset_files)
  │
  └─ Auto-deploy (if enabled)
        └─ Child ModelDeploymentWorkflow on the deployment queue
           (assign 5% → observe → 25% → 50% → 100%, rollback on failure)
```

**Separation of concerns:**
- **Temporal** = orchestration (decides what and when, handles failures, gates, rollback)
- **Argo / Kubeflow Trainer** = training compute; **Katib** = hyperparameter search
- The worker talks to the Argo Server REST API (`internal/argo`) and to the Kubernetes API (`internal/kube`, `internal/kubeflow`) with a namespaced Role, so there's no kubectl in containers

**Verified on local k3s (CPU, Argo backend):** 300 batches → 29,480 transitions (0 legacy, 5 unlinked dropped), 5,000 CQL updates in 149 s, test-track pass rate 0% → gated. The Katib + Kubeflow variant (`use_katib: true`) was verified on the same snapshot: 4 offline trials → best `batch_size=256, cql_alpha≈3.25, learning_rate≈8.7e-4` → train and eval TrainJobs → gated (pass rate 0%). Those rejections are expected: the fleet's data was idle/falling behaviour from a barely-trained policy.

**Measured trade-off:** offline training is not cheaper per update. On the same 3,200 gradient updates (2–4 CPU pod, simulator running on the same VM), online PPO took 51 s and offline CQL 100 s. The benefits are elsewhere — learning from real robot behaviour (no sim-to-real gap), one dataset reused by every run and Katib trial, and per-model data lineage; the costs move to the data path (larger telemetry, storage) and to heavier updates.

**Not supported yet:** fine-tuning from a base model, `reach` in the pipeline, GPU training (the AWS GPU design is in Terraform but not verified). Offline RL is bounded by the fleet's own behaviour — bootstrap a walking policy with manual online training so fleet data becomes useful. Rewards are computed by the simulator (real robots would need rewards from telemetry). Snapshots list the whole prefix; at fleet scale they need date-range filtering.

### Training Algorithms

| Task Type | Algorithm | Learning Type | Script | Environment | Used by |
|-----------|-----------|---------------|--------|-------------|---------|
| `locomotion` | CQL-SAC | Offline RL from fleet experience | `train_offline.py` | none (test track for eval) | Pipeline |
| `locomotion` | PPO | Online RL (on-policy) | `train_locomotion.py` | FleetOS-Humanoid-v2 | Manual / bootstrap |
| `reach` | SAC+HER | Online RL (off-policy, goal-conditioned) | `train_reach.py` | FleetOS-Reach-v1 | Manual |
| `grasp` | SAC+HER | RL + optional scripted demos | `train_grasp.py` | FleetOS-Grasp-v1 | Manual |
| `bc` | Behavior Cloning | Supervised (expert demos) | `train_bc.py` | (eval only) | Manual |
| `autoencoder` | Autoencoder | Unsupervised (observations) | `train_autoencoder.py` | N/A | Manual |

### Fleet Experience Data

Every telemetry packet carries a training context, and the processor writes it as NDJSON:

```
Simulator (50 Hz physics, 10 Hz telemetry)
  → per record: exact 86-dim policy observation, action actually applied, reward,
    episode_id, step, exploration flag (terminal record on every fall)
  → EXPLORATION_FRACTION (default 0.2, deterministic per robot) adds N(0, EXPLORATION_STD=0.1) noise
  → Kafka → Processor (internal/experience), 100 transitions per robot per batch
  → s3://fleetos-telemetry/training_data/v2/YYYY/MM/DD/HH/<robot_id>/transitions_<ns>.ndjson
```

Records from before format v2 (directly under `training_data/YYYY/…`, no observation) are never used for training. `train_bc.py` and `train_autoencoder.py` still read the old `experience/` prefix, which nothing writes anymore.

### Custom Training Environments

Three Gymnasium environments using `lab_room.xml`:

| Environment | Obs Space | Action | Reward | Max Steps |
|-------------|-----------|--------|--------|-----------|
| FleetOS-Humanoid-v2 | Box(86,) | 25-dim joint targets | forward_vel + 1.0 - ctrl_cost | 1000 |
| FleetOS-Reach-v1 | Dict{obs(86), goal(3), goal(3)} | 25-dim joint targets | Sparse: -1/0 (HER) | 200 |
| FleetOS-Grasp-v1 | Dict{obs(86), goal(3), goal(3)} | 25-dim joint targets | Sparse: -1/0 (HER) | 300 |

All share the same 86-dim unified observation format (`observation.py`), same physics (0.003s × 7 sub-steps), and same lab environment.

### Training Commands

```bash
# Locomotion (PPO in MuJoCo, from scratch) — manual bootstrap, not the pipeline
python train_locomotion.py --job-id job-001 --timesteps 2000000

# Reaching (SAC+HER)
python train_reach.py --job-id reach-001 --timesteps 500000

# Grasping (SAC+HER with scripted demonstrations)
python train_grasp.py --job-id grasp-001 --timesteps 1000000 --generate-demos 1000

# Offline RL (CQL) from a dataset snapshot — normally run by the pipeline
python train_offline.py --job-id offline-001 \
  --dataset-manifest datasets/<job>/manifest.json --gradient-steps 20000 --s3-endpoint minio:9000

# Behavior cloning (supervised, from high-reward experience)
python train_bc.py --job-id bc-001 --task-type locomotion \
  --demo-source experience --min-reward 2.0 --s3-endpoint minio:9000

# Behavior cloning (from expert policy rollouts)
python train_bc.py --job-id bc-002 --task-type locomotion \
  --demo-source expert --expert-model training/job-001/policy.zip

# Fine-tune RL from BC pre-trained model
python train_locomotion.py --job-id job-003 --timesteps 500000 \
  --base-model training/bc-001/policy.zip

# Autoencoder (unsupervised representation learning)
python train_autoencoder.py --job-id ae-001 --experience-prefix experience/ \
  --latent-dim 32 --epochs 200 --s3-endpoint minio:9000
```

### Infrastructure (Terraform + Helm)

| Resource | Spec | Purpose |
|----------|------|---------|
| GPU node group | g5.2xlarge, 0-8, scale-from-zero (designed, not verified) | Training jobs on AWS |
| Kubeflow namespace | Created by Terraform; Trainer/Katib must be installed separately | ML workloads isolation |
| IAM role (IRSA) | S3 read/write on models + training-data | Secure artifact access |
| S3 lifecycle | training-data bucket → Glacier 30d → delete 180d (Terraform; fleet experience currently lives in the telemetry bucket under `training_data/v2/`) | Cost management |
| Katib (Helm `kubeflow.hpo`) | TPE by default, 4 trials × 2 parallel, 2,000 offline updates + 5 test-track episodes per trial (local budget) | Hyperparameter search |

## Testing

```bash
# All Go tests (11 suites)
go test -race ./internal/... ./test/...

# Python SDK tests (12 tests)
cd sdk/python && python3 -m pytest test_fleetos.py -v

# FAISS vector search tests (14 tests)
cd inference && python3 -m pytest test_vector_search.py -v

# Coverage
go test -race -coverprofile=coverage.out ./...
```

### Test Coverage

| Package | Tests | What's tested |
|---------|-------|---------------|
| `internal/api` | 5 | JSON serialization, input validation, error paths |
| `internal/auth` | 12 | JWT, API keys, RBAC, OAuth2/OIDC validation |
| `internal/command` | 7 | All 12 command matchers, fallback, custom registry |
| `internal/config` | 5 | Env loading, defaults, invalid input warnings |
| `internal/ingestion` | 2 | Handler creation, Kafka consumer |
| `internal/middleware` | 10 | Logging, CORS, rate limiting (fail-closed), usage metering |
| `internal/service` | 17 | Robot CRUD, fleet metrics, semantic commands, billing |
| `internal/simulator` | 18 | Physics, fleet creation, protobuf, benchmarks |
| `internal/store` | 7 | Redis, Postgres, S3 key format, interface compliance |
| `test/integration` | 15 | HTTP endpoints, auth, inference mock, WebSocket, gRPC |
| `sdk/python` | 12 | All SDK methods against mock server |
| `inference` | 14 | Vector search, embeddings, FAISS index |

## Project Structure

```
platform/
+-- cmd/                        # Service entry points
|   +-- api/                    # REST API + WebSocket + admin console
|   +-- ingestion/              # gRPC telemetry + Kafka command dispatch
|   +-- processor/              # Kafka consumer → Postgres/Redis/S3 + metrics aggregation
|   +-- worker/                 # Temporal worker (4 task queues: commands, deployments, billing, webhooks)
+-- admin-web/                  # React admin console (Vite + React 19 + TypeScript)
+-- internal/                   # Core Go packages
|   +-- api/                    # Thin HTTP handlers (handler, billing, admin, agent, model, etc.)
|   +-- service/                # Business logic layer
|   |   +-- robot_service.go    # Robot CRUD, fleet metrics
|   |   +-- inference_service.go # Per-robot model lookup + inference
|   |   +-- billing_service.go  # Pricing tiers, Temporal billing integration
|   |   +-- billing_temporal_service.go # Billing workflows (invoices, tier changes)
|   |   +-- admin_service.go    # Tenant + API key management
|   |   +-- model_service.go    # Model registry lifecycle
|   +-- billing/                # Shared pricing types (breaks import cycles)
|   +-- command/                # Semantic command registry (strategy pattern)
|   +-- auth/                   # JWT, API keys (DB-backed), OAuth2/OIDC, RBAC
|   +-- middleware/             # Rate limiting, quota, metrics, logging, CORS, tracing
|   +-- store/                  # PostgreSQL, Redis, S3, ClickHouse (interfaces + implementations)
|   +-- temporal/               # Client, workflows, activities, Kafka ack bridge
|   |   +-- workflows/          # Command, deployment, billing, training pipeline, webhook
|   |   +-- activities/         # Command, deployment, billing, training, inference, webhook
|   +-- config/                 # Environment-based configuration
+-- inference/                   # Python inference service (SB3 policy serving)
|   +-- server.py               # SB3 PPO/SAC policy serving + instruction bias
+-- analytics/                  # Spark analytics
|   +-- telemetry_analytics.py  # PySpark batch job (S3 -> ClickHouse)
|   +-- clickhouse-init.sql     # OLAP schema
|   +-- run.sh                  # Spark submit script
+-- sdk/                        # Developer SDKs
|   +-- python/                 # Python SDK + tests
|   +-- typescript/             # TypeScript SDK
+-- ros2_bridge/                # ROS 2 bridge node
+-- proto/                      # Protobuf definitions
+-- migrations/                 # PostgreSQL DDL
+-- deploy/                     # Deployment configs
|   +-- docker/                 # Dockerfiles (multi-stage, non-root)
|   +-- helm/fleetos/           # Helm chart + Istio manifests
|   +-- terraform/              # AWS (EKS, RDS, ElastiCache, S3)
+-- observability/              # Prometheus, Grafana, alerting
+-- docs/                       # OpenAPI spec, architecture diagrams
+-- test/integration/           # Integration tests
+-- docker-compose.yml          # Local dev (17 services)
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Go 1.26, gRPC, Protocol Buffers, net/http |
| Orchestration | Temporal (durable command workflows, deployment pipelines) |
| AI/ML | Python, PyTorch, SB3 (CQL-SAC offline; PPO/SAC online), FAISS, MuJoCo (simulation + evaluation) |
| Messaging | Kafka (Confluent), gRPC bidirectional streaming |
| Storage | PostgreSQL 16, Redis 7, MinIO (S3), ClickHouse |
| Analytics | Apache Spark (PySpark), ClickHouse (OLAP) |
| API | REST, WebSocket, OpenAPI 3.1, OAuth2/OIDC, JWT |
| SDKs | TypeScript, Python (zero-dependency) |
| Robotics | ROS 2 (Humble), sensor_msgs, geometry_msgs |
| Infrastructure | Kubernetes, Docker, Helm, Terraform (AWS), Istio |
| Observability | Prometheus, Grafana, OpenTelemetry, PagerDuty |
| CI/CD | GitHub Actions, ArgoCD, GHCR |
| Security | mTLS (Istio), SSE (S3), encryption at rest (RDS/Redis), non-root containers |

## Deployment

### Docker Compose (local dev)

```bash
# Both stacks from repo root
../start.sh

# Platform only
docker compose up -d
```

### Kubernetes (Helm + Istio)

```bash
helm upgrade --install fleetos deploy/helm/fleetos \
  --namespace fleetos --create-namespace \
  -f deploy/helm/fleetos/values.yaml \
  --set istio.enabled=true
```

### Terraform (AWS)

```bash
cd deploy/terraform
terraform init && terraform plan && terraform apply
```

Provisions: VPC (3 AZ), EKS (general + GPU + Kafka nodes), RDS PostgreSQL (Multi-AZ, encrypted), ElastiCache Redis (HA, encrypted), S3 (SSE, lifecycle, versioning).

## Observability

| Tool | URL | Purpose |
|------|-----|---------|
| Web Playground | http://localhost:5173 | Interactive robot dashboard |
| Grafana | http://localhost:3000 | Metrics dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Time-series metrics |
| Temporal UI | http://localhost:8233 | Workflow execution history |
| Spark UI | http://localhost:8088 | Spark job monitoring |
| MinIO Console | http://localhost:9001 | S3 object browser (fleetos/fleetos123) |
| ClickHouse | http://localhost:8123 | Direct SQL analytics |

## License

MIT
