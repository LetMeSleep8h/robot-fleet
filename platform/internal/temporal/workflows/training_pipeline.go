package workflows

import (
	"fmt"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	temporalpkg "github.com/dimuthu/robot-fleet/internal/temporal"
	"github.com/dimuthu/robot-fleet/internal/temporal/activities"
)

// TrainingPipelineInput is the input for TrainingPipelineWorkflow.
type TrainingPipelineInput struct {
	JobID          string `json:"job_id"`
	Timesteps      int    `json:"timesteps"`       // default: 2000000
	EnvID          string `json:"env_id"`          // empty = WorkflowTemplate default
	TaskType       string `json:"task_type"`       // "locomotion" or "reach"
	Backend        string `json:"backend"`         // "argo" (default) or "kubeflow"
	UseKatib       bool   `json:"use_katib"`       // Katib HPO first; implies the kubeflow backend
	AutoDeploy     bool   `json:"auto_deploy"`     // auto canary deploy if evaluation passes
	BaseModel      string `json:"base_model"`      // S3 key for fine-tuning, empty = from scratch
	FromExperience bool   `json:"from_experience"` // include collected experience
}

// TrainingPipelineResult is the output of TrainingPipelineWorkflow.
type TrainingPipelineResult struct {
	JobID       string  `json:"job_id"`
	ModelID     string  `json:"model_id"`
	Phase       string  `json:"phase"` // "trained", "registered", "deployed", "failed", "gated"
	MeanReward  float64 `json:"mean_reward"`
	PassRate    float64 `json:"pass_rate"`
	Improvement float64 `json:"improvement_pct"`
	GateReason  string  `json:"gate_reason,omitempty"`
	// Set when a Katib search chose the hyperparameters.
	HPOExperiment string            `json:"hpo_experiment,omitempty"`
	BestParams    map[string]string `json:"best_params,omitempty"`
}

// SignalRetrain is the signal name to trigger retraining.
const SignalRetrain = "retrain"

// TrainingPipelineWorkflow orchestrates the full ML lifecycle:
// [Katib hyperparameter search] → train + evaluate (Argo workflow or Kubeflow
// TrainJobs) → gate on the evaluation results → register → canary deploy.
//
// Can be triggered by:
// 1. Manual API call (POST /api/v1/locomotion/jobs)
// 2. Signal from monitoring when success_rate drops below threshold
// 3. Cron schedule (weekly retraining)
func TrainingPipelineWorkflow(ctx workflow.Context, input TrainingPipelineInput) (*TrainingPipelineResult, error) {
	logger := workflow.GetLogger(ctx)

	if input.Timesteps == 0 {
		input.Timesteps = 2_000_000
	}
	if input.TaskType == "" {
		input.TaskType = "locomotion"
	}

	algorithm := "ppo"
	switch input.TaskType {
	case "reach":
		algorithm = "sac"
	case "grasp":
		algorithm = "sac"
	case "offline":
		algorithm = "cql"
	case "bc":
		algorithm = "bc"
	case "autoencoder":
		algorithm = "autoencoder"
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

	// Step 1: Check experience data availability
	experiencePrefix := ""
	if input.FromExperience {
		var expStats activities.CollectExperienceOutput
		err := workflow.ExecuteActivity(actCtx, "CollectExperienceStats", activities.CollectExperienceInput{
			S3Bucket: "fleetos-models",
			Prefix:   "experience/",
		}).Get(ctx, &expStats)
		if err == nil && expStats.TransitionCount > 1000 {
			experiencePrefix = "experience/"
			logger.Info("experience data available", "transitions", expStats.TransitionCount)
		}
	}

	// Step 2 (optional): Katib hyperparameter search. Katib runs on Kubeflow,
	// so the final training run uses the kubeflow backend too.
	if input.UseKatib {
		input.Backend = activities.BackendKubeflow
		var search activities.HyperparameterSearchOutput
		if err := workflow.ExecuteActivity(longActCtx, "RunHyperparameterSearch", activities.HyperparameterSearchInput{
			JobID:    input.JobID,
			TaskType: input.TaskType,
			EnvID:    input.EnvID,
		}).Get(ctx, &search); err != nil {
			result.Phase = "failed"
			return result, fmt.Errorf("hyperparameter search failed: %w", err)
		}
		result.HPOExperiment = search.Experiment
		result.BestParams = search.BestParams
		logger.Info("katib search complete", "experiment", search.Experiment, "best_params", search.BestParams, "objective", search.BestObjective)
	}

	// Step 3: Train + evaluate on the chosen backend (blocks until done)
	logger.Info("submitting training", "job_id", input.JobID, "timesteps", input.Timesteps, "backend", input.Backend)

	var trainResult activities.SubmitTrainingRunOutput
	err := workflow.ExecuteActivity(longActCtx, "SubmitTrainingRun", activities.SubmitTrainingRunInput{
		JobID:           input.JobID,
		Timesteps:       input.Timesteps,
		EnvID:           input.EnvID,
		BaseModel:       input.BaseModel,
		FromExperience:  experiencePrefix,
		TaskType:        input.TaskType,
		Algorithm:       algorithm,
		Backend:         input.Backend,
		Hyperparameters: result.BestParams,
	}).Get(ctx, &trainResult)
	if err != nil {
		result.Phase = "failed"
		return result, fmt.Errorf("training submission failed: %w", err)
	}

	logger.Info("training finished", "run_id", trainResult.RunID, "model_path", trainResult.ModelPath)

	// Step 4: Gate on this job's evaluation results vs the deployed baseline
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

	// Gate: only proceed if model passes evaluation threshold
	if !evalResult.PassesGate {
		result.Phase = "gated"
		result.GateReason = evalResult.GateReason
		logger.Warn("model did not pass evaluation gate", "reason", evalResult.GateReason)
		_ = workflow.ExecuteActivity(actCtx, "NotifyTrainingComplete", input.JobID, false).Get(ctx, nil)
		return result, nil
	}

	// Step 5: Register model in registry
	var modelID string
	version := fmt.Sprintf("v%d", workflow.Now(ctx).Unix())
	name := "ppo-lab"
	switch input.TaskType {
	case "reach":
		name = "sac-reach"
	case "grasp":
		name = "sac-grasp"
	case "offline":
		name = "cql-offline"
	case "bc":
		name = "bc-pretrain"
	case "autoencoder":
		name = "autoencoder"
	}
	err = workflow.ExecuteActivity(actCtx, "RegisterTrainedModel", activities.RegisterTrainedModelInput{
		Name:            name,
		Version:         version,
		ArtifactURL:     trainResult.ModelPath,
		MeanReward:      evalResult.MeanReward,
		PassRate:        evalResult.PassRate,
		Environment:     input.EnvID,
		TaskType:        input.TaskType,
		Algorithm:       algorithm,
		Backend:         backendName(input.Backend),
		Hyperparameters: result.BestParams,
		HPOExperiment:   result.HPOExperiment,
	}).Get(ctx, &modelID)
	if err != nil {
		result.Phase = "failed"
		return result, fmt.Errorf("model registration failed: %w", err)
	}

	result.ModelID = modelID
	result.Phase = "registered"
	logger.Info("model registered", "model_id", modelID)

	// Step 6: Auto-deploy via canary (if enabled)
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
			result.Phase = "registered" // deployed failed but model is still in registry
			logger.Warn("canary deployment failed or rolled back", "model_id", modelID)
		} else {
			result.Phase = "deployed"
			logger.Info("model deployed via canary", "model_id", modelID)
		}
	}

	_ = workflow.ExecuteActivity(actCtx, "NotifyTrainingComplete", modelID, result.Phase == "deployed").Get(ctx, nil)

	return result, nil
}

// backendName normalizes the training backend for the registry record.
func backendName(b string) string {
	if b == "" {
		return activities.BackendArgo
	}
	return b
}
