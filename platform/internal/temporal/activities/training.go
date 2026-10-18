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

// OfflineTrainingTemplate is the Argo WorkflowTemplate that trains offline
// from a dataset snapshot (no simulator), uploads training/<job-id>/policy.zip,
// then evaluates it on the MuJoCo test track into evaluations/<job-id>/.
const OfflineTrainingTemplate = "fleetos-offline-training"

// EvalEnvID is the MuJoCo "test track" every candidate is evaluated on.
const EvalEnvID = "FleetOS-Humanoid-v2"

// argoHyperparameterParams maps tunable hyperparameters to template parameters.
var argoHyperparameterParams = map[string]string{
	"learning_rate": "learning-rate",
	"cql_alpha":     "cql-alpha",
	"batch_size":    "batch-size",
}

// WorkflowRunner submits and tracks Argo workflows (implemented by *argo.Client).
type WorkflowRunner interface {
	SubmitTemplate(ctx context.Context, template string, params, labels map[string]string) (string, error)
	GetWorkflowStatus(ctx context.Context, name string) (argo.Status, error)
}

// ArtifactStore reads and writes training artifacts in the models bucket.
type ArtifactStore interface {
	Get(ctx context.Context, key string) ([]byte, error)
	Put(ctx context.Context, key string, data []byte, contentType string) error
}

// DatasetLister lists fleet experience batches in the telemetry bucket.
type DatasetLister interface {
	List(ctx context.Context, prefix string, maxKeys int) ([]string, error)
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
	Models        store.ModelRepository
	Robots        store.RobotRepository
	Runner        WorkflowRunner
	Kubeflow      KubeflowRunner
	HPO           HPOSettings
	ModelsBucket  string // bucket holding training artifacts (for eval model paths)
	Artifacts     ArtifactStore
	Dataset       DatasetLister // fleet experience (telemetry bucket, training_data/)
	DatasetBucket string
	Gate          modelgate.Policy
	PollInterval  time.Duration
}

// ModelArtifactKey is where a training job uploads its policy.
func ModelArtifactKey(jobID string) string { return fmt.Sprintf("training/%s/policy.zip", jobID) }

// EvalResultsKey is where a training job's evaluation step uploads its results.
func EvalResultsKey(jobID string) string {
	return fmt.Sprintf("evaluations/%s/eval_results.json", jobID)
}

// SubmitTrainingRunInput is the input for SubmitTrainingRun.
type SubmitTrainingRunInput struct {
	JobID           string            `json:"job_id"`
	DatasetManifest string            `json:"dataset_manifest"`          // from SnapshotDataset
	GradientSteps   int               `json:"gradient_steps"`            // 0 = trainer default
	Backend         string            `json:"backend"`                   // BackendArgo (default) or BackendKubeflow
	Hyperparameters map[string]string `json:"hyperparameters,omitempty"` // e.g. Katib's best trial
}

// SubmitTrainingRunOutput is the result of a completed training run.
type SubmitTrainingRunOutput struct {
	RunID     string `json:"run_id"` // Argo workflow name
	Status    string `json:"status"` // final Argo phase
	ModelPath string `json:"model_path"`
}

// SubmitTrainingRun trains a policy offline from the dataset snapshot (no
// simulator) and evaluates it on the test track, on the chosen backend. It
// blocks until done; retries resume the same run instead of duplicating it.
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

// submitArgoRun runs OfflineTrainingTemplate, heartbeating the workflow name
// so a retried attempt resumes polling the same run.
func (a *TrainingActivities) submitArgoRun(ctx context.Context, input SubmitTrainingRunInput) (*SubmitTrainingRunOutput, error) {
	if a.Runner == nil {
		return nil, temporal.NewNonRetryableApplicationError(
			"training runner not configured (set ARGO_SERVER_URL)", ErrTypeTrainingNotConfigured, nil)
	}
	params, err := argoTrainingParams(input)
	if err != nil {
		return nil, err
	}

	var name string
	if activity.HasHeartbeatDetails(ctx) && activity.GetHeartbeatDetails(ctx, &name) == nil && name != "" {
		slog.Info("resuming training run", "job_id", input.JobID, "workflow", name)
	} else {
		name, err = a.Runner.SubmitTemplate(ctx, OfflineTrainingTemplate, params, map[string]string{LabelJobID: input.JobID})
		if err != nil {
			return nil, fmt.Errorf("submit training run %s: %w", input.JobID, err)
		}
		slog.Info("training run submitted", "job_id", input.JobID, "workflow", name, "dataset", input.DatasetManifest)
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

func argoTrainingParams(input SubmitTrainingRunInput) (map[string]string, error) {
	if input.DatasetManifest == "" {
		return nil, temporal.NewNonRetryableApplicationError(
			"training run needs a dataset manifest (run SnapshotDataset first)", ErrTypeUnsupportedTraining, nil)
	}
	params := map[string]string{"job-id": input.JobID, "dataset-manifest": input.DatasetManifest}
	if input.GradientSteps > 0 {
		params["gradient-steps"] = strconv.Itoa(input.GradientSteps)
	}
	for name, value := range input.Hyperparameters {
		param, ok := argoHyperparameterParams[name]
		if !ok {
			return nil, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("unknown hyperparameter %q", name), ErrTypeUnsupportedTraining, nil)
		}
		params[param] = value
	}
	return params, nil
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
	// Data lineage: the fleet-experience snapshot the model was trained on.
	DatasetManifest string `json:"dataset_manifest,omitempty"`
	DatasetFiles    int    `json:"dataset_files,omitempty"`
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
	if input.DatasetManifest != "" {
		model.Metrics["dataset_manifest"] = input.DatasetManifest
		model.Metrics["dataset_files"] = input.DatasetFiles
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
