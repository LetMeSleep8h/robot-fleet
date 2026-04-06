#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 [up|down|restart|logs|ps|status] [--platform] [--playground] [--k8s]"
  echo ""
  echo "Commands:"
  echo "  up        Start services (default)"
  echo "  down      Stop services"
  echo "  restart   Restart services"
  echo "  logs      Tail logs"
  echo "  ps/status Show running services"
  echo ""
  echo "Flags:"
  echo "  --platform    Only platform stack"
  echo "  --playground  Only playground stack"
  echo "  --k8s         Use Kubernetes (kind + ArgoCD) instead of Docker Compose"
  echo "  (no flag)     Both stacks"
  echo ""
  echo "Examples:"
  echo "  $0                  # Docker Compose: start everything"
  echo "  $0 --k8s            # K8s: full stack via Helm + ArgoCD"
  echo "  $0 down --k8s       # K8s: tear down kind cluster"
  echo "  $0 logs --k8s       # K8s: tail all fleetos pod logs"
  exit 1
}

CMD="up"
PLATFORM=false
PLAYGROUND=false
K8S=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    up|down|restart|logs|ps|status) CMD="$1" ;;
    --platform)   PLATFORM=true ;;
    --playground) PLAYGROUND=true ;;
    --k8s)        K8S=true ;;
    -h|--help)    usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

# Default: both stacks
if ! $PLATFORM && ! $PLAYGROUND; then
  PLATFORM=true
  PLAYGROUND=true
fi

# ============================================================
# Docker Compose mode
# ============================================================
run_compose() {
  local dir="$1"
  local name="$2"
  shift 2

  echo "==> $name: docker compose $*"
  docker compose -f "$dir/docker-compose.yml" "$@"
}

compose_mode() {
  case "$CMD" in
    up)
      $PLATFORM   && run_compose "$SCRIPT_DIR/platform"   "Platform"   up -d
      $PLAYGROUND && run_compose "$SCRIPT_DIR/playground"  "Playground" up -d
      ;;
    down)
      $PLAYGROUND && run_compose "$SCRIPT_DIR/playground"  "Playground" down
      $PLATFORM   && run_compose "$SCRIPT_DIR/platform"    "Platform"   down
      ;;
    restart)
      $PLAYGROUND && run_compose "$SCRIPT_DIR/playground"  "Playground" down
      $PLATFORM   && run_compose "$SCRIPT_DIR/platform"    "Platform"   down
      $PLATFORM   && run_compose "$SCRIPT_DIR/platform"    "Platform"   up -d
      $PLAYGROUND && run_compose "$SCRIPT_DIR/playground"   "Playground" up -d
      ;;
    logs)
      if $PLATFORM && $PLAYGROUND; then
        docker compose -f "$SCRIPT_DIR/platform/docker-compose.yml" \
          -f "$SCRIPT_DIR/playground/docker-compose.yml" logs -f
      elif $PLATFORM; then
        run_compose "$SCRIPT_DIR/platform" "Platform" logs -f
      else
        run_compose "$SCRIPT_DIR/playground" "Playground" logs -f
      fi
      ;;
    ps|status)
      $PLATFORM   && run_compose "$SCRIPT_DIR/platform"   "Platform"   ps
      $PLAYGROUND && run_compose "$SCRIPT_DIR/playground"  "Playground" ps
      ;;
  esac
}

# ============================================================
# Kubernetes mode (kind + ArgoCD + Helm)
# ============================================================
k8s_mode() {
  case "$CMD" in
    up)
      echo "==> K8s: Starting full stack via Helm + ArgoCD"
      cd "$SCRIPT_DIR/platform" && make k8s-up
      ;;
    down)
      echo "==> K8s: Tearing down kind cluster"
      cd "$SCRIPT_DIR/platform" && make k8s-down
      ;;
    restart)
      echo "==> K8s: Restarting (destroy + recreate)"
      cd "$SCRIPT_DIR/platform" && make k8s-down && make k8s-up
      ;;
    logs)
      local svc="api"
      if ! $PLATFORM && $PLAYGROUND; then
        svc="simulator"
      fi
      echo "==> K8s: Tailing logs (SVC=$svc, change with: make k8s-logs SVC=<name>)"
      kubectl logs -n fleetos -l app.kubernetes.io/instance=fleetos --all-containers --tail=100 -f 2>/dev/null || \
        kubectl logs -n fleetos --all-containers --tail=100 -f
      ;;
    ps|status)
      echo "==> K8s: Cluster status"
      cd "$SCRIPT_DIR/platform" && make k8s-status
      ;;
  esac
}

# ============================================================
# Dispatch
# ============================================================
if $K8S; then
  k8s_mode
else
  compose_mode
fi
