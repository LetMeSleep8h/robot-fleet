#!/usr/bin/env bash
# Sets up a local kind cluster with the full FleetOS stack.
# Everything runs in K8s — infra (Postgres, Redis, Kafka, MinIO, Temporal),
# platform services, and playground. ArgoCD manages the Helm release.
#
# Prerequisites: kind, kubectl, helm, docker
# Usage:
#   scripts/k8s-setup.sh           # Full setup
#   scripts/k8s-setup.sh --images  # Rebuild + reload images only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PLATFORM_DIR")"
K8S_DIR="$PLATFORM_DIR/deploy/k8s"
HELM_DIR="$PLATFORM_DIR/deploy/helm/fleetos"
CLUSTER_NAME="${CLUSTER_NAME:-fleetos}"
ARGOCD_VERSION="${ARGOCD_VERSION:-v2.13.3}"
ARGO_WF_VERSION="${ARGO_WF_VERSION:-v3.5.5}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[k8s]${NC} $*"; }
warn() { echo -e "${YELLOW}[k8s]${NC} $*"; }
err()  { echo -e "${RED}[k8s]${NC} $*" >&2; }
step() { echo -e "\n${CYAN}--- $* ---${NC}"; }

# --- Preflight ---
preflight() {
    step "Preflight checks"
    for cmd in kind kubectl helm docker; do
        if ! command -v "$cmd" &>/dev/null; then
            err "$cmd is required but not installed"
            exit 1
        fi
        log "  $cmd OK"
    done
}

# --- Build Helm dependencies ---
helm_deps() {
    step "Helm dependencies"
    cd "$HELM_DIR"
    if [ -d "charts" ] && ls charts/*.tgz >/dev/null 2>&1; then
        log "Subchart tarballs already present"
    else
        log "Fetching Bitnami subcharts (postgresql, redis, kafka, minio)..."
        helm dependency build .
    fi
    cd "$PLATFORM_DIR"
}

# --- Create kind cluster ---
create_cluster() {
    step "Kind cluster"
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        log "Cluster '$CLUSTER_NAME' already exists"
    else
        # Inject the actual repo root path into kind config
        local kind_config
        kind_config=$(mktemp)
        sed "s|REPO_ROOT_PLACEHOLDER|${REPO_ROOT}|g" "$K8S_DIR/kind-config.yaml" > "$kind_config"

        log "Creating kind cluster '$CLUSTER_NAME'..."
        kind create cluster \
            --name "$CLUSTER_NAME" \
            --config "$kind_config" \
            --wait 120s
        rm -f "$kind_config"
    fi

    kubectl cluster-info --context "kind-${CLUSTER_NAME}" >/dev/null 2>&1
    log "Context: kind-${CLUSTER_NAME}"
}

# --- Build and load images ---
load_images() {
    step "Docker images"
    cd "$PLATFORM_DIR"

    # Platform services
    local platform_images=("ingestion" "api" "processor" "worker")
    for svc in "${platform_images[@]}"; do
        local img="fleetos/${svc}:latest"
        if ! docker image inspect "$img" >/dev/null 2>&1; then
            log "  Building $img..."
            docker build -t "$img" -f "deploy/docker/Dockerfile.${svc}" . 2>&1 | tail -1
        fi
        log "  Loading $img..."
        kind load docker-image "$img" --name "$CLUSTER_NAME"
    done

    # Inference
    local inf_img="fleetos/inference:latest"
    if ! docker image inspect "$inf_img" >/dev/null 2>&1; then
        log "  Building $inf_img..."
        docker build -t "$inf_img" -f "deploy/docker/Dockerfile.inference" . 2>&1 | tail -1
    fi
    log "  Loading $inf_img..."
    kind load docker-image "$inf_img" --name "$CLUSTER_NAME"

    # Playground simulator
    cd "$REPO_ROOT/playground"
    local sim_img="fleetos/simulator:latest"
    if ! docker image inspect "$sim_img" >/dev/null 2>&1; then
        log "  Building $sim_img..."
        docker build -t "$sim_img" -f "deploy/Dockerfile.simulator" . 2>&1 | tail -1
    fi
    log "  Loading $sim_img..."
    kind load docker-image "$sim_img" --name "$CLUSTER_NAME"

    # Playground web
    local web_img="fleetos/web:latest"
    if ! docker image inspect "$web_img" >/dev/null 2>&1; then
        log "  Building $web_img..."
        docker build -t "$web_img" -f "deploy/Dockerfile.web" . 2>&1 | tail -1
    fi
    log "  Loading $web_img..."
    kind load docker-image "$web_img" --name "$CLUSTER_NAME"

    # Training
    cd "$PLATFORM_DIR"
    local train_img="fleetos/training:latest"
    if ! docker image inspect "$train_img" >/dev/null 2>&1; then
        log "  Building $train_img..."
        docker build -t "$train_img" ./training 2>&1 | tail -1
    fi
    log "  Loading $train_img..."
    kind load docker-image "$train_img" --name "$CLUSTER_NAME"

    # External images that need pre-loading into kind.
    # Bitnami subcharts use :latest (specific version tags removed from Docker Hub).
    # Kafka/MinIO use standalone (non-Bitnami) images.
    local external_images=(
        "docker.io/bitnami/postgresql:latest"
        "docker.io/bitnami/redis:latest"
        "confluentinc/cp-kafka:7.6.0"
        "confluentinc/cp-zookeeper:7.6.0"
        "cgr.dev/chainguard/minio:latest"
        "cgr.dev/chainguard/minio-client:latest"
        "temporalio/auto-setup:1.25"
        "busybox:1.36"
    )
    for img in "${external_images[@]}"; do
        if ! docker image inspect "$img" >/dev/null 2>&1; then
            log "  Pulling $img..."
            docker pull "$img"
        fi
        log "  Loading $img..."
        # kind load docker-image fails on multi-platform OCI images (--all-platforms flag),
        # so pipe through docker save → ctr import directly.
        if ! docker save "$img" | docker exec -i "${CLUSTER_NAME}-control-plane" \
            ctr --namespace=k8s.io images import - 2>/dev/null; then
            warn "  ctr import failed for $img, trying kind load..."
            kind load docker-image "$img" --name "$CLUSTER_NAME"
        fi
    done
}

# --- Install ArgoCD ---
install_argocd() {
    step "ArgoCD"
    if kubectl get deployment argocd-server -n argocd >/dev/null 2>&1; then
        log "ArgoCD already installed"
    else
        log "Installing ArgoCD ${ARGOCD_VERSION}..."
        kubectl create namespace argocd 2>/dev/null || true
        kubectl apply -n argocd \
            -f "https://raw.githubusercontent.com/argoproj/argo-cd/${ARGOCD_VERSION}/manifests/install.yaml"

        log "Waiting for ArgoCD server..."
        kubectl wait --for=condition=available deployment/argocd-server \
            -n argocd --timeout=180s

        kubectl rollout status statefulset/argocd-application-controller \
            -n argocd --timeout=180s 2>/dev/null || true
    fi

    # Expose ArgoCD UI via NodePort
    kubectl apply -f "$K8S_DIR/argocd/argocd-nodeport.yaml"

    # Patch repo-server to mount host repo
    log "Patching argocd-repo-server with host repo mount..."
    kubectl patch deployment argocd-repo-server -n argocd \
        --type strategic \
        --patch-file "$K8S_DIR/argocd/repo-server-patch.yaml"

    kubectl rollout status deployment/argocd-repo-server -n argocd --timeout=120s

    ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
        -o jsonpath="{.data.password}" 2>/dev/null | base64 -d 2>/dev/null || echo "pending")
    log "ArgoCD admin password: $ARGOCD_PASSWORD"
}

# --- Deploy via ArgoCD ---
deploy_argocd() {
    step "Deploying FleetOS via ArgoCD"

    # Apply AppProject
    kubectl apply -f "$K8S_DIR/argocd/appproject.yaml"

    # Apply Application (ArgoCD syncs the Helm chart)
    kubectl apply -f "$K8S_DIR/argocd/application-local.yaml"

    log "Waiting for ArgoCD to sync..."
    local retries=0
    local max_retries=60
    while [ $retries -lt $max_retries ]; do
        local status
        status=$(kubectl get application fleetos-platform -n argocd \
            -o jsonpath='{.status.sync.status}' 2>/dev/null || echo "Unknown")
        local health
        health=$(kubectl get application fleetos-platform -n argocd \
            -o jsonpath='{.status.health.status}' 2>/dev/null || echo "Unknown")

        if [ "$status" = "Synced" ] && [ "$health" = "Healthy" ]; then
            log "ArgoCD sync complete: $status / $health"
            break
        fi

        if [ $((retries % 10)) -eq 0 ]; then
            log "  Sync: $status, Health: $health (waiting...)"
        fi

        sleep 5
        retries=$((retries + 1))
    done

    if [ $retries -ge $max_retries ]; then
        warn "ArgoCD sync timed out. Check ArgoCD UI at http://localhost:30443"
        warn "Status:"
        kubectl get application fleetos-platform -n argocd -o yaml 2>/dev/null | tail -20
    fi
}

# --- Install Argo Workflows (for training) ---
install_argo_workflows() {
    step "Argo Workflows"
    if kubectl get deployment workflow-controller -n argo >/dev/null 2>&1; then
        log "Argo Workflows already installed"
    else
        log "Installing Argo Workflows ${ARGO_WF_VERSION}..."
        kubectl create namespace argo 2>/dev/null || true
        kubectl apply -n argo \
            -f "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_WF_VERSION}/quick-start-minimal.yaml"

        kubectl wait --for=condition=available deployment/workflow-controller \
            -n argo --timeout=120s 2>/dev/null || true
    fi

    # Expose Argo UI via NodePort
    kubectl patch svc argo-server -n argo --type merge \
        -p '{"spec":{"type":"NodePort","ports":[{"port":2746,"targetPort":2746,"nodePort":30746}]}}'

    # Apply training secrets and workflow templates in argo namespace
    kubectl apply -f "$K8S_DIR/training-secrets.yaml"
    if [ -d "$K8S_DIR/workflows" ] && ls "$K8S_DIR/workflows"/*.yaml >/dev/null 2>&1; then
        kubectl apply -n argo -f "$K8S_DIR/workflows/"
    fi
}

# --- Summary ---
print_summary() {
    step "FleetOS Platform Ready"
    echo ""
    log "  Cluster:       $CLUSTER_NAME"
    log "  Context:       kind-${CLUSTER_NAME}"
    echo ""
    log "  ArgoCD UI:     http://localhost:30080"
    log "  ArgoCD Login:  admin / ${ARGOCD_PASSWORD:-<make k8s-argocd-password>}"
    log "  Argo Wf UI:    https://localhost:30746"
    echo ""
    log "  Services:"
    log "    Web UI:      http://localhost:30173"
    log "    API:         http://localhost:30880"
    log "    Ingestion:   localhost:30051 (gRPC)"
    log "    Inference:   http://localhost:30081"
    log "    Simulator:   http://localhost:30085"
    log "    Prometheus:  http://localhost:30090"
    log "    Grafana:     http://localhost:30030  (admin/admin)"
    echo ""
    log "  Commands:"
    log "    make k8s-status          # Pod and service status"
    log "    make k8s-logs SVC=api    # Tail logs"
    log "    make k8s-sync            # Force ArgoCD re-sync"
    log "    make k8s-train           # Submit training workflow"
    log "    make k8s-down            # Delete cluster"
    echo ""
}

# --- Main ---
preflight

case "${1:-}" in
    --images)
        load_images
        ;;
    *)
        helm_deps
        create_cluster
        load_images
        install_argocd
        deploy_argocd
        install_argo_workflows
        print_summary
        ;;
esac
