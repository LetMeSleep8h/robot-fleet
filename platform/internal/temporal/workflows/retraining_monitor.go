package workflows

import (
	"time"

	"go.temporal.io/sdk/workflow"
)

// RetrainingMonitorInput is the input for RetrainingMonitorWorkflow.
type RetrainingMonitorInput struct {
	Threshold float64 `json:"threshold"` // e.g. 0.7
	TaskType  string  `json:"task_type"` // "locomotion" or "reach"
}

// RetrainingMonitorWorkflow periodically checks if the deployed model's performance
// has dropped below threshold and triggers retraining if needed. Uses continue-as-new
// to avoid unbounded history growth.
func RetrainingMonitorWorkflow(ctx workflow.Context, input RetrainingMonitorInput) error {
	logger := workflow.GetLogger(ctx)

	if input.Threshold == 0 {
		input.Threshold = 0.7
	}
	if input.TaskType == "" {
		input.TaskType = "locomotion"
	}

	actCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 5 * time.Minute,
	})

	var needed bool
	err := workflow.ExecuteActivity(actCtx, "CheckRetrainingNeeded", input.Threshold).Get(ctx, &needed)
	if err != nil {
		logger.Error("failed to check retraining need", "error", err)
		return workflow.NewContinueAsNewError(ctx, RetrainingMonitorWorkflow, input)
	}

	if needed {
		logger.Info("retraining triggered", "threshold", input.Threshold, "task_type", input.TaskType)
		childCtx := workflow.WithChildOptions(ctx, workflow.ChildWorkflowOptions{})
		err = workflow.ExecuteChildWorkflow(childCtx, TrainingPipelineWorkflow, TrainingPipelineInput{
			TaskType:   input.TaskType,
			AutoDeploy: true,
		}).Get(ctx, nil)
		if err != nil {
			logger.Error("retraining failed", "error", err)
		}
	}

	return workflow.NewContinueAsNewError(ctx, RetrainingMonitorWorkflow, input)
}
