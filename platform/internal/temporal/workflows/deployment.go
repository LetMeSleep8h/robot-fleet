package workflows

import (
	"fmt"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/dimuthu/robot-fleet/internal/temporal/activities"
)

// DeploymentWorkflowInput is the input for ModelDeploymentWorkflow.
type DeploymentWorkflowInput struct {
	ModelID string `json:"model_id"`
}

// DeploymentWorkflowResult is the output of ModelDeploymentWorkflow.
type DeploymentWorkflowResult struct {
	ModelID string `json:"model_id"`
	Phase   string `json:"phase"`   // "deployed" or "rolled_back"
	Percent int    `json:"percent"` // final rollout percentage
}

// SignalManualRollback is the signal name for operator-initiated rollback.
const SignalManualRollback = "manual-rollback"

var canaryStages = []int{5, 25, 50, 100}

const observationPeriod = 5 * time.Minute

// ModelDeploymentWorkflow orchestrates a progressive canary deployment:
// staged → canary: assign 5% → observe → judge → 25% → observe → judge → 50% →
// observe → judge → 100% (deployed). Any failed stage rolls back.
// Supports manual rollback via signal at any stage.
func ModelDeploymentWorkflow(ctx workflow.Context, input DeploymentWorkflowInput) (*DeploymentWorkflowResult, error) {
	actCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 30 * time.Second,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 3,
		},
	})

	// Start canary: validate model, find baseline, set status
	var baseline activities.StartCanaryOutput
	err := workflow.ExecuteActivity(actCtx, "StartCanary", input.ModelID).Get(ctx, &baseline)
	if err != nil {
		return nil, err
	}

	rollbackCh := workflow.GetSignalChannel(ctx, SignalManualRollback)
	rollback := func(percent int) (*DeploymentWorkflowResult, error) {
		_ = workflow.ExecuteActivity(actCtx, "RollbackModel", activities.RollbackInput{CanaryModelID: input.ModelID, BaselineModelID: baseline.BaselineModelID}).Get(ctx, nil) // best-effort; result reports the rollback
		return &DeploymentWorkflowResult{ModelID: input.ModelID, Phase: "rolled_back", Percent: percent}, nil
	}

	for _, percent := range canaryStages {
		if percent >= 100 {
			break // the final stage is FinalizeDeployment below
		}

		// Put this stage's robots on the canary model before judging it.
		if err := workflow.ExecuteActivity(actCtx, "ExpandRollout", activities.ExpandRolloutInput{
			ModelID: input.ModelID,
			Percent: percent,
		}).Get(ctx, nil); err != nil {
			workflow.GetLogger(ctx).Error("expand rollout failed", "percent", percent, "error", err)
			return rollback(percent)
		}

		// Wait for the observation window OR a manual rollback signal.
		timerCtx, timerCancel := workflow.WithCancel(ctx)
		timerFuture := workflow.NewTimer(timerCtx, observationPeriod)

		manualRollback := false
		sel := workflow.NewSelector(ctx)
		sel.AddReceive(rollbackCh, func(ch workflow.ReceiveChannel, more bool) {
			var reason string
			ch.Receive(ctx, &reason)
			manualRollback = true
			timerCancel()
		})
		sel.AddFuture(timerFuture, func(f workflow.Future) {})
		sel.Select(ctx)

		if manualRollback {
			return rollback(percent)
		}

		// Judge the canary on metrics gathered while it served this stage.
		var metricsOK bool
		err := workflow.ExecuteActivity(actCtx, "CompareCanaryMetrics", activities.CompareMetricsInput{
			CanaryModelID:   input.ModelID,
			BaselineModelID: baseline.BaselineModelID,
		}).Get(ctx, &metricsOK)
		if err != nil || !metricsOK {
			return rollback(percent)
		}
	}

	// Full deployment: set model to deployed, archive baseline, assign all robots.
	if err := workflow.ExecuteActivity(actCtx, "FinalizeDeployment", activities.FinalizeInput{
		ModelID:         input.ModelID,
		BaselineModelID: baseline.BaselineModelID,
	}).Get(ctx, nil); err != nil {
		return nil, fmt.Errorf("finalize deployment of %s: %w", input.ModelID, err)
	}

	return &DeploymentWorkflowResult{ModelID: input.ModelID, Phase: "deployed", Percent: 100}, nil
}
