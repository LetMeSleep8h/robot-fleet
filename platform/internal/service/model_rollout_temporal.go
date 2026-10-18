package service

import (
	"context"
	"fmt"

	"go.temporal.io/sdk/client"

	temporalpkg "github.com/dimuthu/robot-fleet/internal/temporal"
	"github.com/dimuthu/robot-fleet/internal/temporal/workflows"
)

// TemporalCanaryStarter starts ModelDeploymentWorkflow on the deployment queue.
type TemporalCanaryStarter struct {
	temporal client.Client
}

// NewTemporalCanaryStarter creates a CanaryStarter backed by Temporal.
func NewTemporalCanaryStarter(tc client.Client) *TemporalCanaryStarter {
	return &TemporalCanaryStarter{temporal: tc}
}

// StartCanary starts (or refuses to duplicate) the model's rollout workflow.
func (s *TemporalCanaryStarter) StartCanary(ctx context.Context, modelID string) (string, error) {
	opts := client.StartWorkflowOptions{
		ID:        "deploy-" + modelID,
		TaskQueue: temporalpkg.TaskQueueDeployment,
		// One rollout per model: a repeated deploy request must not attach to,
		// and appear to restart, a rollout that is already in progress.
		WorkflowExecutionErrorWhenAlreadyStarted: true,
	}
	run, err := s.temporal.ExecuteWorkflow(ctx, opts, workflows.ModelDeploymentWorkflow,
		workflows.DeploymentWorkflowInput{ModelID: modelID})
	if err != nil {
		return "", fmt.Errorf("start deployment workflow for %s: %w", modelID, err)
	}
	return run.GetID(), nil
}
