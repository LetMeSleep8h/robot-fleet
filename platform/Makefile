.PHONY: all build test proto docker-build docker-push compose-up compose-down helm-install lint clean

# Variables
REGISTRY ?= fleetos
TAG ?= latest
SERVICES := ingestion api processor worker

all: proto build

# --- Build ---
build:
	@echo "Building all services..."
	@for svc in $(SERVICES); do \
		echo "  Building $$svc..."; \
		go build -o bin/$$svc ./cmd/$$svc; \
	done

build-%:
	go build -o bin/$* ./cmd/$*

# --- Protobuf ---
proto:
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		-I. proto/telemetry.proto proto/api.proto proto/simulation.proto
	mv proto/*.pb.go internal/telemetry/ 2>/dev/null || true
	mv proto/*_grpc.pb.go internal/telemetry/ 2>/dev/null || true

# --- Test ---
test:
	go test -v -race ./...

test-cover:
	go test -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# --- Lint ---
lint:
	golangci-lint run ./...

# --- Docker ---
docker-build:
	@for svc in $(SERVICES); do \
		echo "Building docker image: $(REGISTRY)/$$svc:$(TAG)"; \
		docker build -t $(REGISTRY)/$$svc:$(TAG) -f deploy/docker/Dockerfile.$$svc .; \
	done

docker-push:
	@for svc in $(SERVICES); do \
		docker push $(REGISTRY)/$$svc:$(TAG); \
	done

# --- Docker Compose (local dev) ---
compose-up:
	docker compose up -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

# --- Helm ---
helm-deps:
	helm dependency build deploy/helm/fleetos

helm-install: helm-deps
	helm upgrade --install fleetos deploy/helm/fleetos \
		--namespace fleetos --create-namespace \
		-f deploy/helm/fleetos/values.yaml

helm-install-local: helm-deps
	helm upgrade --install fleetos deploy/helm/fleetos \
		--namespace fleetos --create-namespace \
		-f deploy/helm/fleetos/values.yaml \
		-f deploy/helm/fleetos/values-local.yaml

helm-template: helm-deps
	helm template fleetos deploy/helm/fleetos \
		-f deploy/helm/fleetos/values.yaml \
		-f deploy/helm/fleetos/values-local.yaml

helm-uninstall:
	helm uninstall fleetos --namespace fleetos

# --- Run locally ---
run-ingestion:
	go run ./cmd/ingestion

run-api:
	go run ./cmd/api

run-simulate-telemetry:
	go run scripts/simulate_telemetry.go -target localhost:50051 -robots 5

# --- Database ---
db-migrate:
	psql "$(POSTGRES_DSN)" -f migrations/001_init.sql

# --- Kubernetes (local kind + ArgoCD + Argo Workflows) ---
K8S_CLUSTER ?= fleetos
TASK ?= reach
TIMESTEPS ?= 100000
JOB_ID ?= $(TASK)-$(shell date +%s)
SVC ?= api

# Full stack on K8s with ArgoCD (infra + platform + playground)
k8s-up:
	CLUSTER_NAME=$(K8S_CLUSTER) scripts/k8s-setup.sh

k8s-down:
	kind delete cluster --name $(K8S_CLUSTER)

# Rebuild a service image and reload into kind
k8s-reload:
	docker build -t fleetos/$(SVC):latest -f deploy/docker/Dockerfile.$(SVC) .
	kind load docker-image fleetos/$(SVC):latest --name $(K8S_CLUSTER)
	kubectl rollout restart deployment/fleetos-$(SVC) -n fleetos

# Rebuild and reload images only (no cluster recreate)
k8s-images:
	CLUSTER_NAME=$(K8S_CLUSTER) scripts/k8s-setup.sh --images

# Force ArgoCD to re-sync the Helm release
k8s-sync:
	kubectl -n argocd patch application fleetos-platform --type merge \
		-p '{"operation":{"initiatedBy":{"username":"cli"},"sync":{"revision":"HEAD"}}}'

k8s-training-image:
	docker build -t fleetos/training:latest ./training

k8s-train:
	@echo "Submitting $(TASK) training: $(TIMESTEPS) steps, job=$(JOB_ID)"
	kubectl create -f - <<EOF
	apiVersion: argoproj.io/v1alpha1
	kind: Workflow
	metadata:
	  generateName: fleetos-$(TASK)-
	spec:
	  entrypoint: train-and-eval
	  templates:
	    - name: train-and-eval
	      dag:
	        tasks:
	          - name: train
	            template: train-task
	          - name: evaluate
	            template: eval-task
	            dependencies: [train]
	    - name: train-task
	      container:
	        image: fleetos/training:latest
	        command: [python]
	        args:
	          - train_$(TASK).py
	          - --job-id=$(JOB_ID)
	          - --timesteps=$(TIMESTEPS)
	          - --output-dir=/tmp/output
	        envFrom:
	          - secretRef:
	              name: fleetos-training
	    - name: eval-task
	      container:
	        image: fleetos/training:latest
	        command: [python]
	        args:
	          - evaluate_$(TASK).py
	          - --model-path=/tmp/output/policy.zip
	          - --output-dir=/tmp/eval
	        envFrom:
	          - secretRef:
	              name: fleetos-training
	EOF

k8s-status:
	@echo "=== Namespaces ===" && kubectl get ns
	@echo ""
	@echo "=== FleetOS Pods ===" && kubectl get pods -n fleetos -o wide 2>/dev/null || true
	@echo ""
	@echo "=== FleetOS Services ===" && kubectl get svc -n fleetos 2>/dev/null || true
	@echo ""
	@echo "=== ArgoCD ===" && kubectl get pods -n argocd 2>/dev/null || true
	@echo ""
	@echo "=== Argo Workflows ===" && kubectl get workflows -n argo 2>/dev/null || true

k8s-logs:
	kubectl logs -n fleetos deployment/fleetos-$(SVC) --tail=100 -f

k8s-restart:
	kubectl rollout restart deployment/fleetos-$(SVC) -n fleetos

k8s-argocd-password:
	@kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo

# --- Clean ---
clean:
	rm -rf bin/ coverage.out coverage.html

# --- Terraform ---
tf-init:
	cd deploy/terraform && terraform init

tf-plan:
	cd deploy/terraform && terraform plan

tf-apply:
	cd deploy/terraform && terraform apply
