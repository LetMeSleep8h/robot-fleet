package service

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"
	"go.temporal.io/api/enums/v1"
	"go.temporal.io/api/workflow/v1"
	"go.temporal.io/api/workflowservice/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/mocks"

	temporalpkg "github.com/dimuthu/robot-fleet/internal/temporal"
	"github.com/dimuthu/robot-fleet/internal/temporal/workflows"
)

func TestStartPipeline_StartsTrainingWorkflow(t *testing.T) {
	tc := &mocks.Client{}
	run := &mocks.WorkflowRun{}
	run.On("GetID").Return("pipeline-x")
	var gotOpts client.StartWorkflowOptions
	var gotInput workflows.TrainingPipelineInput
	tc.On("ExecuteWorkflow", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Run(func(args mock.Arguments) {
			gotOpts = args.Get(1).(client.StartWorkflowOptions)
			gotInput = args.Get(3).(workflows.TrainingPipelineInput)
		}).Return(run, nil)

	svc := NewPipelineService(tc)
	res, err := svc.StartPipeline(context.Background(), "tenant-dev", PipelineRequest{
		TaskType: "locomotion", GradientSteps: 20000, MinTransitions: 5000, MaxDatasetFiles: 200,
		Backend: "kubeflow", UseKatib: true, AutoDeploy: true,
	})
	if err != nil {
		t.Fatalf("StartPipeline: %v", err)
	}
	if gotOpts.TaskQueue != temporalpkg.TaskQueueTraining || !strings.HasPrefix(gotOpts.ID, "pipeline-") {
		t.Errorf("unexpected start options %+v", gotOpts)
	}
	if gotInput.JobID == "" || gotInput.TaskType != "locomotion" || gotInput.GradientSteps != 20000 ||
		gotInput.MinTransitions != 5000 || gotInput.MaxDatasetFiles != 200 ||
		gotInput.Backend != "kubeflow" || !gotInput.UseKatib || !gotInput.AutoDeploy {
		t.Errorf("unexpected workflow input %+v", gotInput)
	}
	if res.WorkflowID != "pipeline-x" || res.JobID != gotInput.JobID || res.Status != PipelineStatusRunning {
		t.Errorf("unexpected result %+v", res)
	}
}

func TestStartPipeline_Validation(t *testing.T) {
	tests := []struct {
		name string
		req  PipelineRequest
	}{
		{"unknown task", PipelineRequest{TaskType: "grasp"}},
		{"reach has no fleet dataset yet", PipelineRequest{TaskType: "reach"}},
		{"unknown backend", PipelineRequest{TaskType: "locomotion", Backend: "sagemaker"}},
		{"katib on argo", PipelineRequest{TaskType: "locomotion", Backend: "argo", UseKatib: true}},
		{"negative gradient steps", PipelineRequest{TaskType: "locomotion", GradientSteps: -1}},
		{"negative min transitions", PipelineRequest{TaskType: "locomotion", MinTransitions: -1}},
		{"negative max files", PipelineRequest{TaskType: "locomotion", MaxDatasetFiles: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := NewPipelineService(&mocks.Client{})
			if _, err := svc.StartPipeline(context.Background(), "t", tt.req); !errors.Is(err, ErrInvalidInput) {
				t.Errorf("err = %v, want ErrInvalidInput", err)
			}
		})
	}
}

func TestStartPipeline_NoTemporalIsUnavailable(t *testing.T) {
	svc := NewPipelineService(nil)
	if _, err := svc.StartPipeline(context.Background(), "t", PipelineRequest{TaskType: "locomotion"}); !errors.Is(err, ErrUnavailable) {
		t.Errorf("err = %v, want ErrUnavailable", err)
	}
}

func TestGetPipeline(t *testing.T) {
	describe := func(status enums.WorkflowExecutionStatus) *workflowservice.DescribeWorkflowExecutionResponse {
		return &workflowservice.DescribeWorkflowExecutionResponse{
			WorkflowExecutionInfo: &workflow.WorkflowExecutionInfo{Status: status},
		}
	}

	t.Run("running", func(t *testing.T) {
		tc := &mocks.Client{}
		tc.On("DescribeWorkflowExecution", mock.Anything, "pipeline-1", "").Return(describe(enums.WORKFLOW_EXECUTION_STATUS_RUNNING), nil)
		res, err := NewPipelineService(tc).GetPipeline(context.Background(), "pipeline-1")
		if err != nil || res.Status != PipelineStatusRunning || res.Result != nil {
			t.Errorf("res=%+v err=%v", res, err)
		}
	})

	t.Run("completed includes result", func(t *testing.T) {
		tc := &mocks.Client{}
		tc.On("DescribeWorkflowExecution", mock.Anything, "pipeline-1", "").Return(describe(enums.WORKFLOW_EXECUTION_STATUS_COMPLETED), nil)
		run := &mocks.WorkflowRun{}
		run.On("Get", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
			*args.Get(1).(*workflows.TrainingPipelineResult) = workflows.TrainingPipelineResult{JobID: "j", Phase: "gated", GateReason: "below"}
		}).Return(nil)
		tc.On("GetWorkflow", mock.Anything, "pipeline-1", "").Return(run)

		res, err := NewPipelineService(tc).GetPipeline(context.Background(), "pipeline-1")
		if err != nil || res.Status != PipelineStatusCompleted || res.Result == nil || res.Result.Phase != "gated" {
			t.Errorf("res=%+v err=%v", res, err)
		}
	})

	t.Run("unknown workflow", func(t *testing.T) {
		tc := &mocks.Client{}
		tc.On("DescribeWorkflowExecution", mock.Anything, "nope", "").Return(nil, errors.New("workflow not found"))
		if _, err := NewPipelineService(tc).GetPipeline(context.Background(), "nope"); err == nil {
			t.Error("expected error")
		}
	})
}
