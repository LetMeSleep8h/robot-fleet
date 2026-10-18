package service

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/dimuthu/robot-fleet/internal/store"
)

// Training job status constants.
const (
	TrainingStatusQueued    = "queued"
	TrainingStatusRunning   = "running"
	TrainingStatusCompleted = "completed"
	TrainingStatusFailed    = "failed"
)

// Training defaults.
const (
	DefaultTrainingTimesteps = 1_000_000
	DefaultTrainingAlgorithm = "PPO"
	DefaultTrainingEnv       = "FleetOS-Humanoid-v2" // the env the inference service serves (86-dim obs)
	DefaultTrainingDevice    = "auto"
	DefaultEvalScenarios     = 100
)

// SubmitTrainingJobRequest holds the input for submitting a training job.
type SubmitTrainingJobRequest struct {
	AgentID     string         `json:"agent_id,omitempty"`
	Algorithm   string         `json:"algorithm"`
	Environment string         `json:"environment"`
	Timesteps   int64          `json:"timesteps"`
	Device      string         `json:"device"`
	Config      map[string]any `json:"config"`
}

// SubmitEvalRequest holds the input for submitting a policy evaluation.
type SubmitEvalRequest struct {
	JobID     string `json:"job_id"`
	Scenarios int    `json:"scenarios"`
}

// TrainingCallbackPayload is the payload sent by the training container on completion.
type TrainingCallbackPayload struct {
	JobID          string         `json:"job_id"`
	Status         string         `json:"status"`
	ModelPath      string         `json:"model_path"`
	ArtifactURL    string         `json:"artifact_url,omitempty"`
	Metrics        map[string]any `json:"metrics,omitempty"`
	ErrorMessage   string         `json:"error_message,omitempty"`
	EvalMeanReward float64        `json:"eval_mean_reward,omitempty"`
}

// EvalCallbackPayload is the payload sent by the eval container on completion.
type EvalCallbackPayload struct {
	EvalID          string         `json:"eval_id"`
	Status          string         `json:"status"`
	ScenariosTotal  int            `json:"scenarios_total"`
	ScenariosPassed int            `json:"scenarios_passed"`
	PassRate        float64        `json:"pass_rate"`
	Results         map[string]any `json:"results,omitempty"`
	ErrorMessage    string         `json:"error_message,omitempty"`
}

// JobSubmitter starts training and evaluation runs (Argo workflows in practice).
type JobSubmitter interface {
	SubmitTrainingJob(ctx context.Context, job *store.TrainingJobRecord) error
	SubmitEvalJob(ctx context.Context, eval *store.TrainingEvalRecord, job *store.TrainingJobRecord) error
}

// TrainingService defines the business logic for Cyclotron-style training.
type TrainingService interface {
	SubmitJob(ctx context.Context, tenantID, initiatedBy string, req SubmitTrainingJobRequest) (*store.TrainingJobRecord, error)
	GetJob(ctx context.Context, tenantID, jobID string) (*store.TrainingJobRecord, error)
	ListJobs(ctx context.Context, tenantID, status string, limit int) ([]*store.TrainingJobRecord, error)
	HandleTrainingCallback(ctx context.Context, payload TrainingCallbackPayload) error
	SubmitEval(ctx context.Context, tenantID string, req SubmitEvalRequest) (*store.TrainingEvalRecord, error)
	GetEval(ctx context.Context, tenantID, evalID string) (*store.TrainingEvalRecord, error)
	ListEvals(ctx context.Context, tenantID, jobID string, limit int) ([]*store.TrainingEvalRecord, error)
	HandleEvalCallback(ctx context.Context, payload EvalCallbackPayload) error
}

type trainingService struct {
	repo      store.TrainingRepository
	submitter JobSubmitter
}

// NewTrainingService creates a new training service. A nil submitter means
// training is not configured: submissions fail with ErrUnavailable instead of
// recording jobs that will never run.
func NewTrainingService(repo store.TrainingRepository, submitter JobSubmitter) TrainingService {
	return &trainingService{repo: repo, submitter: submitter}
}

func (s *trainingService) SubmitJob(ctx context.Context, tenantID, initiatedBy string, req SubmitTrainingJobRequest) (*store.TrainingJobRecord, error) {
	if s.submitter == nil {
		return nil, fmt.Errorf("submit training job: training is not configured (set ARGO_SERVER_URL): %w", ErrUnavailable)
	}
	algorithm := strings.ToUpper(req.Algorithm)
	if algorithm == "" {
		algorithm = DefaultTrainingAlgorithm
	}
	if _, err := templatesFor(algorithm); err != nil {
		return nil, fmt.Errorf("submit training job: %w", err)
	}
	env := req.Environment
	if env == "" {
		env = DefaultTrainingEnv
	}
	timesteps := req.Timesteps
	if timesteps <= 0 {
		timesteps = DefaultTrainingTimesteps
	}
	device := req.Device
	if device == "" {
		device = DefaultTrainingDevice
	}
	config := req.Config
	if config == nil {
		config = map[string]any{}
	}

	job := &store.TrainingJobRecord{
		ID:          generateUUID(),
		TenantID:    tenantID,
		AgentID:     req.AgentID,
		Status:      TrainingStatusQueued,
		Algorithm:   algorithm,
		Environment: env,
		Timesteps:   timesteps,
		Device:      device,
		Config:      config,
		Metrics:     map[string]any{},
		InitiatedBy: initiatedBy,
		CreatedAt:   time.Now().UTC(),
	}

	if err := s.repo.CreateTrainingJob(ctx, job); err != nil {
		return nil, fmt.Errorf("create training job: %w", err)
	}

	// The training workflow evaluates the model as its last step and reports
	// under the job's ID, so record that evaluation up front for its callback.
	builtinEval := &store.TrainingEvalRecord{
		ID:             job.ID,
		TenantID:       tenantID,
		JobID:          job.ID,
		Status:         TrainingStatusQueued,
		ScenariosTotal: DefaultEvalScenarios,
		Metrics:        map[string]any{},
		Results:        map[string]any{},
		CreatedAt:      job.CreatedAt,
	}
	if err := s.repo.CreateTrainingEval(ctx, builtinEval); err != nil {
		return nil, fmt.Errorf("create evaluation for training job %s: %w", job.ID, err)
	}

	if err := s.submitter.SubmitTrainingJob(ctx, job); err != nil {
		// Keep the record so the failure is visible on the job.
		_ = s.repo.UpdateTrainingJobCompleted(ctx, job.ID, TrainingStatusFailed, "", err.Error()) // best-effort; failure returned on the job
		job.Status = TrainingStatusFailed
		job.ErrorMessage = err.Error()
		return job, nil
	}
	if err := s.repo.UpdateTrainingJobStatus(ctx, job.ID, TrainingStatusRunning); err != nil {
		return nil, fmt.Errorf("mark training job %s running: %w", job.ID, err)
	}
	job.Status = TrainingStatusRunning

	return job, nil
}

func (s *trainingService) GetJob(ctx context.Context, tenantID, jobID string) (*store.TrainingJobRecord, error) {
	job, err := s.repo.GetTrainingJob(ctx, jobID)
	if err != nil {
		return nil, err
	}
	if job.TenantID != tenantID {
		return nil, ErrNotFound
	}
	return job, nil
}

func (s *trainingService) ListJobs(ctx context.Context, tenantID, status string, limit int) ([]*store.TrainingJobRecord, error) {
	return s.repo.ListTrainingJobs(ctx, tenantID, status, limit)
}

func (s *trainingService) HandleTrainingCallback(ctx context.Context, payload TrainingCallbackPayload) error {
	status := TrainingStatusCompleted
	if payload.Status == "failed" || payload.ErrorMessage != "" {
		status = TrainingStatusFailed
	}

	artifactURL := payload.ArtifactURL
	if artifactURL == "" {
		artifactURL = payload.ModelPath // backward compat
	}

	if err := s.repo.UpdateTrainingJobCompleted(ctx, payload.JobID, status, artifactURL, payload.ErrorMessage); err != nil {
		return fmt.Errorf("update training job on callback: %w", err)
	}

	metrics := payload.Metrics
	if payload.EvalMeanReward != 0 {
		if metrics == nil {
			metrics = map[string]any{}
		}
		metrics["eval_mean_reward"] = payload.EvalMeanReward
	}
	if metrics != nil {
		if err := s.repo.UpdateTrainingJobMetrics(ctx, payload.JobID, metrics); err != nil {
			return fmt.Errorf("update training job metrics on callback: %w", err)
		}
	}

	return nil
}

func (s *trainingService) SubmitEval(ctx context.Context, tenantID string, req SubmitEvalRequest) (*store.TrainingEvalRecord, error) {
	// Verify the training job exists and belongs to the tenant
	job, err := s.GetJob(ctx, tenantID, req.JobID)
	if err != nil {
		return nil, fmt.Errorf("get training job for eval: %w", err)
	}

	if job.Status != TrainingStatusCompleted {
		return nil, fmt.Errorf("training job %s is %s, must be completed before evaluation: %w", req.JobID, job.Status, ErrConflict)
	}
	if s.submitter == nil {
		return nil, fmt.Errorf("submit evaluation: training is not configured: %w", ErrUnavailable)
	}

	scenarios := req.Scenarios
	if scenarios <= 0 {
		scenarios = DefaultEvalScenarios
	}

	eval := &store.TrainingEvalRecord{
		ID:             generateUUID(),
		TenantID:       tenantID,
		JobID:          req.JobID,
		Status:         TrainingStatusQueued,
		ScenariosTotal: scenarios,
		Metrics:        map[string]any{},
		Results:        map[string]any{},
		CreatedAt:      time.Now().UTC(),
	}

	if err := s.repo.CreateTrainingEval(ctx, eval); err != nil {
		return nil, fmt.Errorf("create training eval: %w", err)
	}

	if err := s.submitter.SubmitEvalJob(ctx, eval, job); err != nil {
		_ = s.repo.UpdateTrainingEvalCompleted(ctx, eval.ID, 0, scenarios, 0, nil, err.Error()) // best-effort; failure returned on the eval
		eval.Status = TrainingStatusFailed
		eval.ErrorMessage = err.Error()
		return eval, nil
	}

	return eval, nil
}

func (s *trainingService) GetEval(ctx context.Context, tenantID, evalID string) (*store.TrainingEvalRecord, error) {
	eval, err := s.repo.GetTrainingEval(ctx, evalID)
	if err != nil {
		return nil, err
	}
	if eval.TenantID != tenantID {
		return nil, ErrNotFound
	}
	return eval, nil
}

func (s *trainingService) ListEvals(ctx context.Context, tenantID, jobID string, limit int) ([]*store.TrainingEvalRecord, error) {
	return s.repo.ListTrainingEvals(ctx, tenantID, jobID, limit)
}

func (s *trainingService) HandleEvalCallback(ctx context.Context, payload EvalCallbackPayload) error {
	errorMsg := payload.ErrorMessage
	return s.repo.UpdateTrainingEvalCompleted(
		ctx, payload.EvalID, payload.ScenariosPassed, payload.ScenariosTotal,
		payload.PassRate, payload.Results, errorMsg,
	)
}
