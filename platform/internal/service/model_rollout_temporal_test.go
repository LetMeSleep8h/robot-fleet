package service

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/mock"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/mocks"

	temporalpkg "github.com/dimuthu/robot-fleet/internal/temporal"
	"github.com/dimuthu/robot-fleet/internal/temporal/workflows"
)

func TestTemporalCanaryStarter_StartsDeploymentWorkflow(t *testing.T) {
	tc := &mocks.Client{}
	run := &mocks.WorkflowRun{}
	run.On("GetID").Return("deploy-m1")

	var gotOpts client.StartWorkflowOptions
	var gotInput workflows.DeploymentWorkflowInput
	tc.On("ExecuteWorkflow", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Run(func(args mock.Arguments) {
			gotOpts = args.Get(1).(client.StartWorkflowOptions)
			gotInput = args.Get(3).(workflows.DeploymentWorkflowInput)
		}).
		Return(run, nil)

	id, err := NewTemporalCanaryStarter(tc).StartCanary(context.Background(), "m1")
	if err != nil {
		t.Fatalf("StartCanary: %v", err)
	}
	if id != "deploy-m1" {
		t.Errorf("workflow id = %q", id)
	}
	if gotOpts.ID != "deploy-m1" || gotOpts.TaskQueue != temporalpkg.TaskQueueDeployment {
		t.Errorf("unexpected start options %+v", gotOpts)
	}
	if !gotOpts.WorkflowExecutionErrorWhenAlreadyStarted {
		t.Error("a second deploy of the same model must not silently attach to the running rollout")
	}
	if gotInput.ModelID != "m1" {
		t.Errorf("input = %+v", gotInput)
	}
}

func TestTemporalCanaryStarter_AlreadyRunningIsConflict(t *testing.T) {
	tc := &mocks.Client{}
	tc.On("ExecuteWorkflow", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(nil, errors.New("workflow execution already started"))

	_, err := NewTemporalCanaryStarter(tc).StartCanary(context.Background(), "m1")
	if err == nil {
		t.Fatal("expected error")
	}
}
