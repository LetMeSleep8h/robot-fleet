package activities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"time"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"

	"github.com/dimuthu/robot-fleet/internal/argo"
	"github.com/dimuthu/robot-fleet/internal/modelgate"
	"github.com/dimuthu/robot-fleet/internal/store"
)

// Training run polling.
const (
	DefaultPollInterval      = 15 * time.Second
	MaxConsecutivePollErrors = 10
)

// LabelJobID tags every submitted Argo workflow with its FleetOS job ID.
const LabelJobID = "fleetos.io/job-id"

// Non-retryable error types surfaced to the training pipeline workflow.
const (
	ErrTypeTrainingNotConfigured = "TrainingNotConfigured"
	ErrTypeUnsupportedTraining   = "UnsupportedTraining"
	ErrTypeTrainingRunFailed     = "TrainingRunFailed"
)

// trainingTemplates maps a task type to its Argo WorkflowTemplate. Each
// template trains, uploads training/<job-id>/policy.zip and then evaluates it,
// uploading evaluations/<job-id>/eval_results.json.
var trainingTemplates = map[string]string{
	"locomotion": "fleetos-locomotion-training",
	"reach":      "fleetos-reach-training",
}

// templatesWithEnvParam lists templates that accept an env-id parameter.
var templatesWithEnvParam = map[string]bool{"fleetos-locomotion-training": true}

// WorkflowRunner submits and tracks Argo workflows (implemented by *argo.Client).
type WorkflowRunner interface {
	SubmitTemplate(ctx context.Context, template string, params, labels map[string]string) (string, error)
	GetWorkflowStatus(ctx context.Context, name string) (argo.Status, error)
}

// ArtifactReader reads training artifacts from the models bucket.
type ArtifactReader interface {
	Get(ctx context.Context, key string) ([]byte, error)
}

// Training backends.
const (
	BackendArgo     = "argo"     // Argo WorkflowTemplate (train + evaluate); the default
	BackendKubeflow = "kubeflow" // Kubeflow Trainer TrainJobs (train, then evaluate)
)

// TrainingActivities holds dependencies for ML training pipeline activities.
// A nil Runner/Kubeflow means that backend is not configured: runs fail
// instead of faking success.
type TrainingActivities struct {
	Models       store.ModelRepository
	Robots       store.RobotRepository
	Runner       WorkflowRunner
	Kubeflow     KubeflowRunner
	HPO          HPOSettings
	ModelsBucket string // bucket holding training artifacts (for eval model paths)
	Artifacts    ArtifactReader
	Gate         modelgate.Policy
	PollInterval time.Duration
}

// ModelArtifactKey is where a training job uploads its policy.
func ModelArtifactKey(jobID string) string { return fmt.Sprintf("training/%s/policy.zip", jobID) }

// EvalResultsKey is where a training job's evaluation step uploads its results.
func EvalResultsKey(jobID string) string {
	return fmt.Sprintf("evaluations/%s/eval_results.json", jobID)
}

// CollectExperienceInput is the input for CollectExperienceStats.
type CollectExperienceInput struct {
	S3Bucket string `json:"s3_bucket"`
	Prefix   string `json:"prefix"` // e.g. "experience/"
}

// CollectExperienceOutput reports available experience data.
type CollectExperienceOutput struct {
	TransitionCount int    `json:"transition_count"` // estimated
	DateRange       string `json:"date_range"`
}

// CollectExperienceStats checks how much experience data is available in S3.
// Placeholder: returns a fixed estimate. The training templates cannot consume
// experience data yet, so SubmitTrainingRun rejects runs that ask for it.
func (a *TrainingActivities) CollectExperienceStats(ctx context.Context, input CollectExperienceInput) (*CollectExperienceOutput, error) {
	slog.Info("checking experience data", "bucket", input.S3Bucket, "prefix", input.Prefix)
	return &CollectExperienceOutput{
		TransitionCount: 5000,
		DateRange:       time.Now().AddDate(0, 0, -7).Format("2006-01-02") + " to " + time.Now().Format("2006-01-02"),
	}, nil
}

// SubmitTrainingRunInput is the input for SubmitTrainingRun.
type SubmitTrainingRunInput struct {
	JobID           string            `json:"job_id"`
	Timesteps       int               `json:"timesteps"`
	EnvID           string            `json:"env_id"`
	BaseModel       string            `json:"base_model"`      // S3 key for fine-tuning (not supported yet)
	FromExperience  string            `json:"from_experience"` // S3 prefix for experience data (not supported yet)
	TaskType        string            `json:"task_type"`       // "locomotion" or "reach"
	Algorithm       string            `json:"algorithm"`
	Backend         string            `json:"backend"`                   // BackendArgo (default) or BackendKubeflow
	Hyperparameters map[string]string `json:"hyperparameters,omitempty"` // e.g. Katib's best trial (kubeflow backend)
}

// SubmitTrainingRunOutput is the result of a completed training run.
type SubmitTrainingRunOutput struct {
	RunID     string `json:"run_id"` // Argo workflow name
	Status    string `json:"status"` // final Argo phase
	ModelPath string `json:"model_path"`
}

// SubmitTrainingRun submits the task's Argo WorkflowTemplate and blocks until
// the workflow finishes, heartbeating the workflow name so a retried attempt
// resumes polling the same run instead of submitting a duplicate.
func (a *TrainingActivities) SubmitTrainingRun(ctx context.Context, input SubmitTrainingRunInput) (*SubmitTrainingRunOutput, error) {
	switch input.Backend {
	case "", BackendArgo:
		return a.submitArgoRun(ctx, input)
	case BackendKubeflow:
		return a.submitKubeflowRun(ctx, input)
	default:
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("unknown training backend %q (supported: argo, kubeflow)", input.Backend), ErrTypeUnsupportedTraining, nil)
	}
}

func (a *TrainingActivities) submitArgoRun(ctx context.Context, input SubmitTrainingRunInput) (*SubmitTrainingRunOutput, error) {
	if a.Runner == nil {
		return nil, temporal.NewNonRetryableApplicationError(
			"training runner not configured (set ARGO_SERVER_URL)", ErrTypeTrainingNotConfigured, nil)
	}
	template, err := validateTrainingRun(input)
	if err != nil {
		return nil, err
	}

	var name string
	if activity.HasHeartbeatDetails(ctx) && activity.GetHeartbeatDetails(ctx, &name) == nil && name != "" {
		slog.Info("resuming training run", "job_id", input.JobID, "workflow", name)
	} else {
		params := map[string]string{"job-id": input.JobID}
		if input.Timesteps > 0 {
			params["timesteps"] = strconv.Itoa(input.Timesteps)
		}
		if input.EnvID != "" && templatesWithEnvParam[template] {
			params["env-id"] = input.EnvID
		}
		name, err = a.Runner.SubmitTemplate(ctx, template, params, map[string]string{LabelJobID: input.JobID})
		if err != nil {
			return nil, fmt.Errorf("submit training run %s: %w", input.JobID, err)
		}
		slog.Info("training run submitted", "job_id", input.JobID, "workflow", name, "template", template)
	}
	activity.RecordHeartbeat(ctx, name)

	status, err := a.waitForWorkflow(ctx, name)
	if err != nil {
		return nil, err
	}
	if status.Phase != argo.PhaseSucceeded {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("training workflow %s %s: %s", name, status.Phase, status.Message), ErrTypeTrainingRunFailed, nil)
	}

	return &SubmitTrainingRunOutput{RunID: name, Status: status.Phase, ModelPath: ModelArtifactKey(input.JobID)}, nil
}

func validateTrainingRun(input SubmitTrainingRunInput) (string, error) {
	unsupported := func(msg string) error {
		return temporal.NewNonRetryableApplicationError(msg, ErrTypeUnsupportedTraining, nil)
	}
	template, ok := trainingTemplates[input.TaskType]
	switch {
	case !ok:
		return "", unsupported(fmt.Sprintf("no training template for task type %q (supported: locomotion, reach)", input.TaskType))
	case len(input.Hyperparameters) > 0:
		return "", unsupported("custom hyperparameters require the kubeflow backend")
	case input.BaseModel != "":
		return "", unsupported("fine-tuning from a base model is not supported by the training templates")
	case input.FromExperience != "":
		return "", unsupported("training from collected experience is not supported by the training templates")
	}
	return template, nil
}

// waitForWorkflow polls until the workflow reaches a terminal phase.
func (a *TrainingActivities) waitForWorkflow(ctx context.Context, name string) (argo.Status, error) {
	interval := a.PollInterval
	if interval <= 0 {
		interval = DefaultPollInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	consecutiveErrs := 0
	for {
		status, err := a.Runner.GetWorkflowStatus(ctx, name)
		switch {
		case errors.Is(err, argo.ErrWorkflowNotFound):
			return argo.Status{}, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("training workflow %s disappeared", name), ErrTypeTrainingRunFailed, err)
		case err != nil:
			consecutiveErrs++
			slog.Warn("poll training workflow failed", "workflow", name, "attempt", consecutiveErrs, "error", err)
			if consecutiveErrs >= MaxConsecutivePollErrors {
				return argo.Status{}, fmt.Errorf("poll training workflow %s: %w", name, err)
			}
		default:
			consecutiveErrs = 0
			if status.Done() {
				return status, nil
			}
		}
		activity.RecordHeartbeat(ctx, name)

		select {
		case <-ctx.Done():
			return argo.Status{}, ctx.Err()
		case <-ticker.C:
		}
	}
}

// EvaluateModelInput is the input for EvaluateTrainedModel.
type EvaluateModelInput struct {
	JobID           string `json:"job_id"`
	BaselineModelID string `json:"baseline_model_id"` // empty = currently deployed model, if any
}

// EvaluateModelOutput is the result of model evaluation.
type EvaluateModelOutput struct {
	PassRate       float64 `json:"pass_rate"`
	MeanReward     float64 `json:"mean_reward"`
	BaselineID     string  `json:"baseline_id"`
	BaselineReward float64 `json:"baseline_reward"`
	Improvement    float64 `json:"improvement_pct"`
	PassesGate     bool    `json:"passes_gate"`
	GateReason     string  `json:"gate_reason,omitempty"`
}

// evalResults is the subset of eval_results.json we read.
type evalResults struct {
	PassRate   float64 `json:"pass_rate"`
	MeanReward float64 `json:"mean_reward"`
}

// EvaluateTrainedModel reads the job's evaluation results from the models
// bucket and applies the promotion gate against the baseline model.
func (a *TrainingActivities) EvaluateTrainedModel(ctx context.Context, input EvaluateModelInput) (*EvaluateModelOutput, error) {
	if a.Artifacts == nil {
		return nil, temporal.NewNonRetryableApplicationError(
			"artifact store not configured", ErrTypeTrainingNotConfigured, nil)
	}
	raw, err := a.Artifacts.Get(ctx, EvalResultsKey(input.JobID))
	if err != nil {
		return nil, fmt.Errorf("read eval results for job %s: %w", input.JobID, err)
	}
	var res evalResults
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("parse eval results for job %s: %v", input.JobID, err), ErrTypeTrainingRunFailed, err)
	}

	out := &EvaluateModelOutput{PassRate: res.PassRate, MeanReward: res.MeanReward}
	baseline := a.findBaseline(ctx, input.BaselineModelID)
	var baselineMetrics map[string]any
	if baseline != nil {
		out.BaselineID = baseline.ID
		baselineMetrics = baseline.Metrics
		out.BaselineReward = metricFloat(baseline.Metrics, modelgate.MetricEvalMeanReward)
		if out.BaselineReward > 0 {
			out.Improvement = (out.MeanReward - out.BaselineReward) / out.BaselineReward * 100
		}
	}

	candidate := map[string]any{
		modelgate.MetricEvalPassRate:   res.PassRate,
		modelgate.MetricEvalMeanReward: res.MeanReward,
	}
	if gateErr := a.Gate.Check(candidate, baselineMetrics); gateErr != nil {
		out.GateReason = gateErr.Error()
	} else {
		out.PassesGate = true
	}

	slog.Info("model evaluation complete",
		"job_id", input.JobID,
		"pass_rate", out.PassRate,
		"mean_reward", out.MeanReward,
		"baseline", out.BaselineID,
		"passes_gate", out.PassesGate,
		"reason", out.GateReason,
	)
	return out, nil
}

// findBaseline returns the requested baseline, or the deployed model, or nil.
func (a *TrainingActivities) findBaseline(ctx context.Context, id string) *store.ModelRecord {
	if id != "" {
		if m, err := a.Models.GetModel(ctx, id); err == nil {
			return m
		}
		slog.Warn("baseline model not found, falling back to deployed model", "baseline", id)
	}
	deployed, err := a.Models.ListModels(ctx, "deployed")
	if err != nil || len(deployed) == 0 {
		return nil
	}
	return deployed[0]
}

// RegisterTrainedModelInput is the input for RegisterTrainedModel.
type RegisterTrainedModelInput struct {
	Name        string  `json:"name"`
	Version     string  `json:"version"`
	ArtifactURL string  `json:"artifact_url"`
	MeanReward  float64 `json:"mean_reward"`
	PassRate    float64 `json:"pass_rate"`
	Environment string  `json:"environment"`
	TaskType    string  `json:"task_type"`
	Algorithm   string  `json:"algorithm"`
	// Provenance: how the model was produced.
	Backend         string            `json:"backend"`
	Hyperparameters map[string]string `json:"hyperparameters,omitempty"`
	HPOExperiment   string            `json:"hpo_experiment,omitempty"`
}

// RegisterTrainedModel registers a trained model in the model registry as staged.
func (a *TrainingActivities) RegisterTrainedModel(ctx context.Context, input RegisterTrainedModelInput) (string, error) {
	id := fmt.Sprintf("%s-%s", input.Name, input.Version)

	model := &store.ModelRecord{
		ID:          id,
		Name:        input.Name,
		Version:     input.Version,
		ArtifactURL: input.ArtifactURL,
		Status:      "staged",
		Metrics: map[string]any{
			modelgate.MetricEvalMeanReward: input.MeanReward,
			modelgate.MetricEvalPassRate:   input.PassRate,
			"environment":                  input.Environment,
			"task_type":                    input.TaskType,
			"algorithm":                    input.Algorithm,
			"backend":                      input.Backend,
			"trained_at":                   time.Now().UTC().Format(time.RFC3339),
		},
		CreatedAt: time.Now().UTC(),
	}
	if len(input.Hyperparameters) > 0 {
		model.Metrics["hyperparameters"] = input.Hyperparameters
	}
	if input.HPOExperiment != "" {
		model.Metrics["hpo_experiment"] = input.HPOExperiment
	}

	if err := a.Models.RegisterModel(ctx, model); err != nil {
		return "", fmt.Errorf("register trained model: %w", err)
	}

	slog.Info("trained model registered", "id", id, "artifact", input.ArtifactURL)
	return id, nil
}

// CheckRetrainingNeeded checks if the current deployed model's success_rate
// has dropped below threshold, indicating retraining is needed.
func (a *TrainingActivities) CheckRetrainingNeeded(ctx context.Context, threshold float64) (bool, error) {
	deployed, err := a.Models.ListModels(ctx, "deployed")
	if err != nil || len(deployed) == 0 {
		return false, nil
	}

	model := deployed[0]
	successRate := metricFloat(model.Metrics, "success_rate")

	slog.Info("checking retraining need",
		"model", model.ID,
		"success_rate", successRate,
		"threshold", threshold,
	)

	return successRate > 0 && successRate < threshold, nil
}

// NotifyTrainingComplete logs the pipeline outcome. Placeholder for a
// Slack/PagerDuty integration.
func (a *TrainingActivities) NotifyTrainingComplete(ctx context.Context, modelID string, deployed bool) error {
	status := "registered (staged)"
	if deployed {
		status = "deployed via canary"
	}
	slog.Info("training pipeline complete", "model", modelID, "status", status)
	return nil
}
