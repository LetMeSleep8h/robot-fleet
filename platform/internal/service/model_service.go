package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/dimuthu/robot-fleet/internal/modelgate"
	"github.com/dimuthu/robot-fleet/internal/store"
)

// Model lifecycle status constants.
const (
	ModelStatusStaged   = "staged"
	ModelStatusCanary   = "canary"
	ModelStatusDeployed = "deployed"
	ModelStatusArchived = "archived"
)

// ModelDeployedChannel is the Redis pub/sub channel for model deployment events.
const ModelDeployedChannel = "model:deployed"

// DeployStatusCanaryStarted is reported when a progressive rollout begins.
const DeployStatusCanaryStarted = "canary_started"

// ModelRegistryService defines the interface for model lifecycle management.
type ModelRegistryService interface {
	RegisterModel(ctx context.Context, req RegisterModelRequest) (*store.ModelRecord, error)
	GetModel(ctx context.Context, id string) (*store.ModelRecord, error)
	ListModels(ctx context.Context, status string) ([]*store.ModelRecord, error)
	// DeployModel gates a staged model on its evaluation metrics and starts a
	// canary rollout. The rollout, not this call, moves the model to deployed.
	DeployModel(ctx context.Context, id string) (*DeployResult, error)
	// PromoteModel marks a model deployed and notifies inference to load it.
	// Called by the canary workflow once the rollout reaches 100%.
	PromoteModel(ctx context.Context, id string) error
	ArchiveModel(ctx context.Context, id string) error
}

// RegisterModelRequest holds the input for registering a new model.
type RegisterModelRequest struct {
	Name        string         `json:"name"`
	Version     string         `json:"version"`
	ArtifactURL string         `json:"artifact_url"`
	Metrics     map[string]any `json:"metrics"`
}

// DeployResult describes a started rollout.
type DeployResult struct {
	ModelID    string `json:"model_id"`
	Status     string `json:"status"`
	WorkflowID string `json:"workflow_id"`
}

// CanaryStarter starts a progressive canary rollout for a model and returns
// the rollout's workflow ID.
type CanaryStarter interface {
	StartCanary(ctx context.Context, modelID string) (string, error)
}

// ModelServiceOption configures optional dependencies for the model registry service.
type ModelServiceOption func(*modelRegistryService)

// WithCanaryStarter enables deployments via progressive canary rollout.
func WithCanaryStarter(c CanaryStarter) ModelServiceOption {
	return func(s *modelRegistryService) { s.rollout = c }
}

// WithModelGate overrides the default promotion thresholds.
func WithModelGate(p modelgate.Policy) ModelServiceOption {
	return func(s *modelRegistryService) { s.gate = p }
}

type modelRegistryService struct {
	repo    store.ModelRepository
	cache   store.CacheStore
	rollout CanaryStarter // nil = deployments unavailable
	gate    modelgate.Policy
}

// NewModelRegistryService creates a new model registry service.
func NewModelRegistryService(repo store.ModelRepository, cache store.CacheStore, opts ...ModelServiceOption) ModelRegistryService {
	s := &modelRegistryService{repo: repo, cache: cache, gate: modelgate.DefaultPolicy()}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

func (s *modelRegistryService) RegisterModel(ctx context.Context, req RegisterModelRequest) (*store.ModelRecord, error) {
	id := fmt.Sprintf("%s-%s", req.Name, req.Version)
	model := &store.ModelRecord{
		ID:          id,
		Name:        req.Name,
		Version:     req.Version,
		ArtifactURL: req.ArtifactURL,
		Status:      ModelStatusStaged,
		Metrics:     req.Metrics,
		CreatedAt:   time.Now().UTC(),
	}
	if err := s.repo.RegisterModel(ctx, model); err != nil {
		return nil, fmt.Errorf("register model: %w", err)
	}
	return model, nil
}

func (s *modelRegistryService) GetModel(ctx context.Context, id string) (*store.ModelRecord, error) {
	return s.repo.GetModel(ctx, id)
}

func (s *modelRegistryService) ListModels(ctx context.Context, status string) ([]*store.ModelRecord, error) {
	return s.repo.ListModels(ctx, status)
}

func (s *modelRegistryService) DeployModel(ctx context.Context, id string) (*DeployResult, error) {
	model, err := s.repo.GetModel(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("get model %s: %w", id, err)
	}
	if model.Status != ModelStatusStaged {
		return nil, fmt.Errorf("model %s is %s, only staged models can be deployed: %w", id, model.Status, ErrConflict)
	}

	var baseline map[string]any
	if deployed, err := s.repo.ListModels(ctx, ModelStatusDeployed); err == nil && len(deployed) > 0 {
		baseline = deployed[0].Metrics
	}
	if err := s.gate.Check(model.Metrics, baseline); err != nil {
		return nil, fmt.Errorf("deploy model %s: %w", id, err)
	}

	if s.rollout == nil {
		return nil, fmt.Errorf("deploy model %s: canary rollout not configured: %w", id, ErrUnavailable)
	}
	workflowID, err := s.rollout.StartCanary(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("start canary for model %s: %w", id, err)
	}
	slog.Info("canary rollout started", "model", id, "workflow", workflowID)
	return &DeployResult{ModelID: id, Status: DeployStatusCanaryStarted, WorkflowID: workflowID}, nil
}

func (s *modelRegistryService) PromoteModel(ctx context.Context, id string) error {
	model, err := s.repo.GetModel(ctx, id)
	if err != nil {
		return fmt.Errorf("get model %s: %w", id, err)
	}

	if err := s.repo.UpdateModelStatus(ctx, id, ModelStatusDeployed); err != nil {
		return fmt.Errorf("update model status: %w", err)
	}

	payload, err := json.Marshal(map[string]string{
		"model_id":     model.ID,
		"name":         model.Name,
		"version":      model.Version,
		"artifact_url": model.ArtifactURL,
	})
	if err != nil {
		return fmt.Errorf("marshal model deployed event: %w", err)
	}

	// Inference hot-swaps on this event; a missed event leaves robots on the old model.
	if err := s.cache.PublishEvent(ctx, ModelDeployedChannel, payload); err != nil {
		return fmt.Errorf("publish model deployed event: %w", err)
	}
	return nil
}

func (s *modelRegistryService) ArchiveModel(ctx context.Context, id string) error {
	return s.repo.UpdateModelStatus(ctx, id, ModelStatusArchived)
}
