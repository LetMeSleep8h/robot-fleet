package workflows

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"
	"go.temporal.io/sdk/testsuite"

	"github.com/dimuthu/robot-fleet/internal/temporal/activities"
)

func TestModelDeploymentWorkflow_FullRollout(t *testing.T) {
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()

	acts := &activities.DeploymentActivities{}
	env.RegisterActivity(acts)
	env.OnActivity(acts.StartCanary, mock.Anything, mock.Anything).Return(&activities.StartCanaryOutput{
		BaselineModelID: "model-old-v1",
	}, nil)
	env.OnActivity(acts.CompareCanaryMetrics, mock.Anything, mock.Anything).Return(true, nil)
	env.OnActivity(acts.ExpandRollout, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.FinalizeDeployment, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.RollbackModel, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(ModelDeploymentWorkflow, DeploymentWorkflowInput{
		ModelID: "model-new-v2",
	})

	if !env.IsWorkflowCompleted() {
		t.Fatal("workflow should have completed")
	}
	if err := env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}

	var result DeploymentWorkflowResult
	if err := env.GetWorkflowResult(&result); err != nil {
		t.Fatalf("get result: %v", err)
	}
	if result.Phase != "deployed" {
		t.Errorf("expected phase 'deployed', got '%s'", result.Phase)
	}
	if result.Percent != 100 {
		t.Errorf("expected 100%%, got %d%%", result.Percent)
	}
}

func TestModelDeploymentWorkflow_MetricsRollback(t *testing.T) {
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()

	acts := &activities.DeploymentActivities{}
	env.RegisterActivity(acts)
	env.OnActivity(acts.StartCanary, mock.Anything, mock.Anything).Return(&activities.StartCanaryOutput{
		BaselineModelID: "model-old-v1",
	}, nil)
	env.OnActivity(acts.CompareCanaryMetrics, mock.Anything, mock.Anything).Return(false, nil)
	env.OnActivity(acts.RollbackModel, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.ExpandRollout, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.FinalizeDeployment, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(ModelDeploymentWorkflow, DeploymentWorkflowInput{
		ModelID: "model-new-v2",
	})

	if !env.IsWorkflowCompleted() {
		t.Fatal("workflow should have completed")
	}

	var result DeploymentWorkflowResult
	if err := env.GetWorkflowResult(&result); err != nil {
		t.Fatalf("get result: %v", err)
	}
	if result.Phase != "rolled_back" {
		t.Errorf("expected phase 'rolled_back', got '%s'", result.Phase)
	}
}

func TestModelDeploymentWorkflow_ManualRollback(t *testing.T) {
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()

	acts := &activities.DeploymentActivities{}
	env.RegisterActivity(acts)
	env.OnActivity(acts.StartCanary, mock.Anything, mock.Anything).Return(&activities.StartCanaryOutput{
		BaselineModelID: "model-old-v1",
	}, nil)
	env.OnActivity(acts.RollbackModel, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.CompareCanaryMetrics, mock.Anything, mock.Anything).Return(true, nil)
	env.OnActivity(acts.ExpandRollout, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.FinalizeDeployment, mock.Anything, mock.Anything).Return(nil)

	// Send manual rollback signal immediately
	env.RegisterDelayedCallback(func() {
		env.SignalWorkflow(SignalManualRollback, "operator requested")
	}, 0)

	env.ExecuteWorkflow(ModelDeploymentWorkflow, DeploymentWorkflowInput{
		ModelID: "model-new-v2",
	})

	if !env.IsWorkflowCompleted() {
		t.Fatal("workflow should have completed")
	}

	var result DeploymentWorkflowResult
	if err := env.GetWorkflowResult(&result); err != nil {
		t.Fatalf("get result: %v", err)
	}
	if result.Phase != "rolled_back" {
		t.Errorf("expected phase 'rolled_back', got '%s'", result.Phase)
	}
}

func TestModelDeploymentWorkflow_AssignsRobotsBeforeObserving(t *testing.T) {
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()

	var calls []string
	acts := &activities.DeploymentActivities{}
	env.RegisterActivity(acts)
	env.OnActivity(acts.StartCanary, mock.Anything, mock.Anything).Return(&activities.StartCanaryOutput{BaselineModelID: "old"}, nil)
	env.OnActivity(acts.ExpandRollout, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.ExpandRolloutInput) error {
			calls = append(calls, fmt.Sprintf("expand:%d", in.Percent))
			return nil
		})
	env.OnActivity(acts.CompareCanaryMetrics, mock.Anything, mock.Anything).Return(
		func(_ context.Context, _ activities.CompareMetricsInput) (bool, error) {
			calls = append(calls, "compare")
			return true, nil
		})
	env.OnActivity(acts.FinalizeDeployment, mock.Anything, mock.Anything).Return(
		func(_ context.Context, _ activities.FinalizeInput) error {
			calls = append(calls, "finalize")
			return nil
		})
	env.OnActivity(acts.RollbackModel, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(ModelDeploymentWorkflow, DeploymentWorkflowInput{ModelID: "new"})
	if err := env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}

	// Each canary stage must put robots on the model before its metrics are judged.
	want := []string{"expand:5", "compare", "expand:25", "compare", "expand:50", "compare", "finalize"}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Errorf("activity order = %v, want %v", calls, want)
	}
}

func TestModelDeploymentWorkflow_ExpandFailureRollsBack(t *testing.T) {
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()

	rolledBack := false
	acts := &activities.DeploymentActivities{}
	env.RegisterActivity(acts)
	env.OnActivity(acts.StartCanary, mock.Anything, mock.Anything).Return(&activities.StartCanaryOutput{BaselineModelID: "old"}, nil)
	env.OnActivity(acts.ExpandRollout, mock.Anything, mock.Anything).Return(errors.New("postgres down"))
	env.OnActivity(acts.CompareCanaryMetrics, mock.Anything, mock.Anything).Return(true, nil)
	env.OnActivity(acts.FinalizeDeployment, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.RollbackModel, mock.Anything, mock.Anything).Return(
		func(_ context.Context, _ activities.RollbackInput) error { rolledBack = true; return nil })

	env.ExecuteWorkflow(ModelDeploymentWorkflow, DeploymentWorkflowInput{ModelID: "new"})

	var result DeploymentWorkflowResult
	_ = env.GetWorkflowResult(&result)
	if result.Phase != "rolled_back" || !rolledBack {
		t.Errorf("expected rollback after failed expansion, got phase %q rolledBack=%v", result.Phase, rolledBack)
	}
}

func TestModelDeploymentWorkflow_FinalizeFailureIsNotReportedAsDeployed(t *testing.T) {
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()

	acts := &activities.DeploymentActivities{}
	env.RegisterActivity(acts)
	env.OnActivity(acts.StartCanary, mock.Anything, mock.Anything).Return(&activities.StartCanaryOutput{BaselineModelID: "old"}, nil)
	env.OnActivity(acts.ExpandRollout, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(acts.CompareCanaryMetrics, mock.Anything, mock.Anything).Return(true, nil)
	env.OnActivity(acts.FinalizeDeployment, mock.Anything, mock.Anything).Return(errors.New("redis down"))
	env.OnActivity(acts.RollbackModel, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(ModelDeploymentWorkflow, DeploymentWorkflowInput{ModelID: "new"})

	if env.GetWorkflowError() == nil {
		t.Fatal("expected workflow error when finalize fails")
	}
}
