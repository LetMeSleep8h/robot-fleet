package workflows

import (
	"errors"
	"fmt"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	temporalpkg "github.com/dimuthu/robot-fleet/internal/temporal"
	"github.com/dimuthu/robot-fleet/internal/temporal/activities"
)

// Pipeline defaults.
const (
	DefaultMinTransitions = 1_000 // refuse to train on less fleet experience
	OfflineAlgorithm      = "cql-sac"
	offlineModelName      = "cql-fleet"
)

// TrainingPipelineInput is the input for TrainingPipelineWorkflow.
type TrainingPipelineInput struct {
	JobID           string `json:"job_id"`
	TaskType        string `json:"task_type"`         // "locomotion" (recorded on the model)
	Backend         string `json:"backend"`           // "argo" (default) or "kubeflow"
	UseKatib        bool   `json:"use_katib"`         // Katib HPO first; implies the kubeflow backend
	AutoDeploy      bool   `json:"auto_deploy"`       // auto canary deploy if evaluation passes
	GradientSteps   int    `json:"gradient_steps"`    // offline updates; 0 = trainer default
	MinTransitions  int    `json:"min_transitions"`   // 0 = DefaultMinTransitions
	MaxDatasetFiles int    `json:"max_dataset_files"` // newest N experience batches; 0 = all
}

// TrainingPipelineResult is the output of TrainingPipelineWorkflow.
type TrainingPipelineResult struct {
	JobID       string  `json:"job_id"`
	ModelID     string  `json:"model_id"`
	Phase       string  `json:"phase"` // "insufficient_data", "gated", "registered", "deployed", "failed"
	MeanReward  float64 `json:"mean_reward"`
	PassRate    float64 `json:"pass_rate"`
	Improvement float64 `json:"improvement_pct"`
	GateReason  string  `json:"gate_reason,omitempty"`
	// Data lineage: the snapshot of fleet experience this run trained on.
	DatasetManifest string `json:"dataset_manifest,omitempty"`
	DatasetFiles    int    `json:"dataset_files,omitempty"`
	// Set when a Katib search chose the hyperparameters.
	HPOExperiment string            `json:"hpo_experiment,omitempty"`
	BestParams    map[string]string `json:"best_params,omitempty"`
}

// SignalRetrain is the signal name to trigger retraining.
const SignalRetrain = "retrain"

// TrainingPipelineWorkflow learns from the fleet, not from a simulator:
// snapshot fleet experience → [Katib search] → offline CQL training (Argo or
// Kubeflow) → evaluate on the MuJoCo test track → gate → register → canary.
//
// Started by POST /api/v1/training/pipelines.
func TrainingPipelineWorkflow(ctx workflow.Context, input TrainingPipelineInput) (*TrainingPipelineResult, error) {
	logger := workflow.GetLogger(ctx)

	if input.TaskType == "" {
		input.TaskType = "locomotion"
	}
	if input.MinTransitions == 0 {
		input.MinTransitions = DefaultMinTransitions
	}
	if input.JobID == "" {
		input.JobID = fmt.Sprintf("train-%d", workflow.Now(ctx).UnixNano())
	}

	actCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 10 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 3,
		},
	})

	longActCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 4 * time.Hour, // training can take hours
		HeartbeatTimeout:    5 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 2,
		},
	})

	result := &TrainingPipelineResult{JobID: input.JobID}

	// Step 1: Pin the fleet experience this run trains on.
	var snapshot activities.SnapshotDatasetOutput
	if err := workflow.ExecuteActivity(actCtx, "SnapshotDataset", activities.SnapshotDatasetInput{
		JobID:          input.JobID,
		MaxFiles:       input.MaxDatasetFiles,
		MinTransitions: input.MinTransitions,
	}).Get(ctx, &snapshot); err != nil {
		var appErr *temporal.ApplicationError
		if errors.As(err, &appErr) && appErr.Type() == activities.ErrTypeInsufficientData {
			result.Phase = "insufficient_data"
			result.GateReason = appErr.Error()
			logger.Warn("not enough fleet experience to train", "reason", appErr.Error())
			return result, nil
		}
		result.Phase = "failed"
		return result, fmt.Errorf("dataset snapshot failed: %w", err)
	}
	result.DatasetManifest = snapshot.ManifestKey
	result.DatasetFiles = snapshot.Files
	logger.Info("dataset snapshot", "manifest", snapshot.ManifestKey, "files", snapshot.Files, "transitions", snapshot.EstimatedTransitions)

	// Step 2 (optional): Katib hyperparameter search on the same snapshot.
	// Katib runs on Kubeflow, so the final training run does too.
	if input.UseKatib {
		input.Backend = activities.BackendKubeflow
		var search activities.HyperparameterSearchOutput
		if err := workflow.ExecuteActivity(longActCtx, "RunHyperparameterSearch", activities.HyperparameterSearchInput{
			JobID:           input.JobID,
			DatasetManifest: snapshot.ManifestKey,
		}).Get(ctx, &search); err != nil {
			result.Phase = "failed"
			return result, fmt.Errorf("hyperparameter search failed: %w", err)
		}
		result.HPOExperiment = search.Experiment
		result.BestParams = search.BestParams
		logger.Info("katib search complete", "experiment", search.Experiment, "best_params", search.BestParams, "objective", search.BestObjective)
	}

	// Step 3: Offline training + test-track evaluation (blocks until done).
	logger.Info("submitting offline training", "job_id", input.JobID, "backend", input.Backend)
	var trainResult activities.SubmitTrainingRunOutput
	err := workflow.ExecuteActivity(longActCtx, "SubmitTrainingRun", activities.SubmitTrainingRunInput{
		JobID:           input.JobID,
		DatasetManifest: snapshot.ManifestKey,
		GradientSteps:   input.GradientSteps,
		Backend:         input.Backend,
		Hyperparameters: result.BestParams,
	}).Get(ctx, &trainResult)
	if err != nil {
		result.Phase = "failed"
		return result, fmt.Errorf("training submission failed: %w", err)
	}
	logger.Info("training finished", "run_id", trainResult.RunID, "model_path", trainResult.ModelPath)

	// Step 4: Gate on this job's evaluation results vs the deployed baseline.
	var evalResult activities.EvaluateModelOutput
	err = workflow.ExecuteActivity(actCtx, "EvaluateTrainedModel", activities.EvaluateModelInput{
		JobID: input.JobID,
	}).Get(ctx, &evalResult)
	if err != nil {
		result.Phase = "failed"
		return result, fmt.Errorf("evaluation failed: %w", err)
	}

	result.MeanReward = evalResult.MeanReward
	result.PassRate = evalResult.PassRate
	result.Improvement = evalResult.Improvement

	if !evalResult.PassesGate {
		result.Phase = "gated"
		result.GateReason = evalResult.GateReason
		logger.Warn("model did not pass evaluation gate", "reason", evalResult.GateReason)
		_ = workflow.ExecuteActivity(actCtx, "NotifyTrainingComplete", input.JobID, false).Get(ctx, nil) // best-effort notification
		return result, nil
	}

	// Step 5: Register the model with its data lineage.
	var modelID string
	err = workflow.ExecuteActivity(actCtx, "RegisterTrainedModel", activities.RegisterTrainedModelInput{
		Name:            offlineModelName,
		Version:         fmt.Sprintf("v%d", workflow.Now(ctx).Unix()),
		ArtifactURL:     trainResult.ModelPath,
		MeanReward:      evalResult.MeanReward,
		PassRate:        evalResult.PassRate,
		Environment:     activities.EvalEnvID,
		TaskType:        input.TaskType,
		Algorithm:       OfflineAlgorithm,
		Backend:         backendName(input.Backend),
		Hyperparameters: result.BestParams,
		HPOExperiment:   result.HPOExperiment,
		DatasetManifest: snapshot.ManifestKey,
		DatasetFiles:    snapshot.Files,
	}).Get(ctx, &modelID)
	if err != nil {
		result.Phase = "failed"
		return result, fmt.Errorf("model registration failed: %w", err)
	}

	result.ModelID = modelID
	result.Phase = "registered"
	logger.Info("model registered", "model_id", modelID)

	// Step 6: Auto-deploy via canary (if enabled).
	if input.AutoDeploy {
		logger.Info("starting canary deployment", "model_id", modelID)

		childCtx := workflow.WithChildOptions(ctx, workflow.ChildWorkflowOptions{
			WorkflowID: fmt.Sprintf("deploy-%s", modelID),
			// Deployment activities are registered on the deployment worker only.
			TaskQueue: temporalpkg.TaskQueueDeployment,
		})

		var deployResult DeploymentWorkflowResult
		err = workflow.ExecuteChildWorkflow(childCtx, ModelDeploymentWorkflow, DeploymentWorkflowInput{
			ModelID: modelID,
		}).Get(ctx, &deployResult)

		if err != nil || deployResult.Phase == "rolled_back" {
			result.Phase = "registered" // deploy failed but the model stays in the registry
			logger.Warn("canary deployment failed or rolled back", "model_id", modelID)
		} else {
			result.Phase = "deployed"
			logger.Info("model deployed via canary", "model_id", modelID)
		}
	}

	_ = workflow.ExecuteActivity(actCtx, "NotifyTrainingComplete", modelID, result.Phase == "deployed").Get(ctx, nil) // best-effort notification

	return result, nil
}

// backendName normalizes the training backend for the registry record.
func backendName(b string) string {
	if b == "" {
		return activities.BackendArgo
	}
	return b
}
