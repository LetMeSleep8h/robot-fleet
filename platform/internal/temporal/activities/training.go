package activities

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/dimuthu/robot-fleet/internal/store"
)

// TrainingActivities holds dependencies for ML training pipeline activities.
type TrainingActivities struct {
	Models    store.ModelRepository
	Robots    store.RobotRepository
	Submitter *ArgoSubmitter
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
// In production this would query S3 ListObjects; here we return a placeholder
// since the Temporal workflow uses this to decide whether to include --from-experience.
func (a *TrainingActivities) CollectExperienceStats(ctx context.Context, input CollectExperienceInput) (*CollectExperienceOutput, error) {
	slog.Info("checking experience data", "bucket", input.S3Bucket, "prefix", input.Prefix)
	// Placeholder — real implementation queries S3 ListObjects and counts NDJSON lines
	return &CollectExperienceOutput{
		TransitionCount: 5000, // estimated from batch count
		DateRange:       time.Now().AddDate(0, 0, -7).Format("2006-01-02") + " to " + time.Now().Format("2006-01-02"),
	}, nil
}

// SubmitKubeflowRunInput is the input for SubmitKubeflowRun.
type SubmitKubeflowRunInput struct {
	JobID          string `json:"job_id"`
	Timesteps      int    `json:"timesteps"`
	EnvID          string `json:"env_id"`
	BaseModel      string `json:"base_model"`      // S3 key for fine-tuning, empty = from scratch
	FromExperience string `json:"from_experience"` // S3 prefix for experience data
	UseKatib       bool   `json:"use_katib"`       // use HPO via Katib
	TaskType       string `json:"task_type"`        // "locomotion" or "reach"
	Algorithm      string `json:"algorithm"`        // "ppo" or "sac"
}

// SubmitKubeflowRunOutput is the result of submitting a training run.
type SubmitKubeflowRunOutput struct {
	RunID     string `json:"run_id"`
	Status    string `json:"status"` // "submitted", "running", "completed", "failed"
	ModelPath string `json:"model_path"`
}

// SubmitKubeflowRun submits a training job via Argo Workflows (if configured)
// or falls back to stub behavior for local development.
func (a *TrainingActivities) SubmitKubeflowRun(ctx context.Context, input SubmitKubeflowRunInput) (*SubmitKubeflowRunOutput, error) {
	slog.Info("submitting training run",
		"job_id", input.JobID,
		"timesteps", input.Timesteps,
		"env_id", input.EnvID,
		"task_type", input.TaskType,
		"algorithm", input.Algorithm,
		"base_model", input.BaseModel,
		"katib", input.UseKatib,
	)

	artifactURL := fmt.Sprintf("training/%s/policy.zip", input.JobID)

	// Use ArgoSubmitter if configured, otherwise fall back to stub
	if a.Submitter != nil {
		taskType := input.TaskType
		if taskType == "" {
			taskType = "locomotion"
		}

		workflowName, err := a.Submitter.SubmitTraining(ctx, TrainingRunInput{
			JobID:     input.JobID,
			TaskType:  taskType,
			Timesteps: input.Timesteps,
		})
		if err != nil {
			return nil, fmt.Errorf("submit training run: %w", err)
		}

		result, err := a.Submitter.WaitForCompletion(ctx, workflowName)
		if err != nil {
			return nil, fmt.Errorf("training run failed: %w", err)
		}

		if result.ModelPath != "" {
			artifactURL = result.ModelPath
		}

		return &SubmitKubeflowRunOutput{
			RunID:     workflowName,
			Status:    "completed",
			ModelPath: artifactURL,
		}, nil
	}

	// Stub mode: return the expected artifact URL without running actual training
	slog.Info("argo submitter not configured, using stub mode", "job_id", input.JobID)
	return &SubmitKubeflowRunOutput{
		RunID:     input.JobID,
		Status:    "submitted",
		ModelPath: artifactURL,
	}, nil
}

// EvaluateModelInput is the input for EvaluateTrainedModel.
type EvaluateModelInput struct {
	ModelArtifactURL string `json:"model_artifact_url"`
	BaselineModelID  string `json:"baseline_model_id"`
	EvalEpisodes     int    `json:"eval_episodes"`
}

// EvaluateModelOutput is the result of model evaluation.
type EvaluateModelOutput struct {
	MeanReward      float64 `json:"mean_reward"`
	BaselineReward  float64 `json:"baseline_reward"`
	Improvement     float64 `json:"improvement_pct"`
	PassesGate      bool    `json:"passes_gate"`
}

// EvaluateTrainedModel compares the trained model against the current baseline
// using the model registry's eval_mean_reward metric. If no real eval data is
// available (first model), the gate auto-passes.
func (a *TrainingActivities) EvaluateTrainedModel(ctx context.Context, input EvaluateModelInput) (*EvaluateModelOutput, error) {
	output := &EvaluateModelOutput{}

	// Get baseline model metrics
	if input.BaselineModelID != "" {
		baseline, err := a.Models.GetModel(ctx, input.BaselineModelID)
		if err == nil {
			output.BaselineReward = metricFloat(baseline.Metrics, "eval_mean_reward")
		}
	}

	// Try to read eval results from the most recently staged model's metrics
	// (the training callback writes eval_mean_reward into the model record).
	// If no staged model exists yet, this is the first training run — auto-pass.
	staged, err := a.Models.ListModels(ctx, "staged")
	if err == nil && len(staged) > 0 {
		reward := metricFloat(staged[0].Metrics, "eval_mean_reward")
		if reward > 0 {
			output.MeanReward = reward
		}
	}

	// If no real eval data found, first model auto-passes the gate
	if output.MeanReward == 0 && output.BaselineReward == 0 {
		output.PassesGate = true
		slog.Info("first model — auto-passing evaluation gate")
		return output, nil
	}

	if output.BaselineReward > 0 {
		output.Improvement = ((output.MeanReward - output.BaselineReward) / output.BaselineReward) * 100
	}

	// Gate: new model must be within 5% of baseline (early training may be worse)
	output.PassesGate = output.BaselineReward == 0 || output.MeanReward >= output.BaselineReward*0.95

	slog.Info("model evaluation complete",
		"mean_reward", output.MeanReward,
		"baseline_reward", output.BaselineReward,
		"improvement", fmt.Sprintf("%.1f%%", output.Improvement),
		"passes_gate", output.PassesGate,
	)

	return output, nil
}

// RegisterTrainedModelInput is the input for RegisterTrainedModel.
type RegisterTrainedModelInput struct {
	Name        string  `json:"name"`
	Version     string  `json:"version"`
	ArtifactURL string  `json:"artifact_url"`
	MeanReward  float64 `json:"mean_reward"`
	Environment string  `json:"environment"`
	TaskType    string  `json:"task_type"`
	Algorithm   string  `json:"algorithm"`
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
			"eval_mean_reward": input.MeanReward,
			"environment":      input.Environment,
			"task_type":        input.TaskType,
			"algorithm":        input.Algorithm,
			"trained_at":       time.Now().UTC().Format(time.RFC3339),
		},
		CreatedAt: time.Now().UTC(),
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

// NotifyTrainingComplete sends a notification (placeholder for Slack/PagerDuty).
func (a *TrainingActivities) NotifyTrainingComplete(ctx context.Context, modelID string, deployed bool) error {
	status := "registered (staged)"
	if deployed {
		status = "deployed via canary"
	}
	slog.Info("training pipeline complete", "model", modelID, "status", status)

	// Placeholder: POST to Slack webhook, PagerDuty, etc.
	_ = http.DefaultClient
	_ = json.Marshal

	return nil
}
