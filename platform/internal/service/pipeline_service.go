package service

import (
	"context"
	"errors"
	"fmt"

	"go.temporal.io/api/enums/v1"
	"go.temporal.io/api/serviceerror"
	"go.temporal.io/sdk/client"

	temporalpkg "github.com/dimuthu/robot-fleet/internal/temporal"
	"github.com/dimuthu/robot-fleet/internal/temporal/workflows"
)

// Pipeline run statuses reported by the API.
const (
	PipelineStatusRunning   = "running"
	PipelineStatusCompleted = "completed"
	PipelineStatusFailed    = "failed"
)

// Supported pipeline options (see TrainingPipelineWorkflow).
var (
	pipelineTasks    = map[string]bool{"locomotion": true} // tasks with fleet experience to learn from
	pipelineBackends = map[string]bool{"": true, "argo": true, "kubeflow": true}
)

// PipelineRequest starts a training pipeline run, which learns offline from
// the fleet's recorded experience.
type PipelineRequest struct {
	TaskType        string `json:"task_type"`
	GradientSteps   int    `json:"gradient_steps"`    // 0 = trainer default
	MinTransitions  int    `json:"min_transitions"`   // 0 = pipeline default
	MaxDatasetFiles int    `json:"max_dataset_files"` // newest N experience batches; 0 = all
	Backend         string `json:"backend"`           // "argo" (default) or "kubeflow"
	UseKatib        bool   `json:"use_katib"`         // Katib HPO before training (kubeflow)
	AutoDeploy      bool   `json:"auto_deploy"`
}

// PipelineRun describes a pipeline run and, once finished, its result.
type PipelineRun struct {
	WorkflowID string                            `json:"workflow_id"`
	JobID      string                            `json:"job_id,omitempty"`
	Status     string                            `json:"status"`
	Result     *workflows.TrainingPipelineResult `json:"result,omitempty"`
}

// PipelineService starts and reports on Temporal training pipelines.
type PipelineService interface {
	StartPipeline(ctx context.Context, tenantID string, req PipelineRequest) (*PipelineRun, error)
	GetPipeline(ctx context.Context, workflowID string) (*PipelineRun, error)
}

type pipelineService struct {
	temporal client.Client // nil = Temporal disabled
}

// NewPipelineService creates a PipelineService. A nil client makes every call
// return ErrUnavailable.
func NewPipelineService(tc client.Client) PipelineService {
	return &pipelineService{temporal: tc}
}

func (s *pipelineService) StartPipeline(ctx context.Context, tenantID string, req PipelineRequest) (*PipelineRun, error) {
	if err := validatePipeline(req); err != nil {
		return nil, err
	}
	if s.temporal == nil {
		return nil, fmt.Errorf("start pipeline: temporal not configured: %w", ErrUnavailable)
	}

	jobID := "pipe-" + generateUUID()
	run, err := s.temporal.ExecuteWorkflow(ctx, client.StartWorkflowOptions{
		ID:        "pipeline-" + jobID,
		TaskQueue: temporalpkg.TaskQueueTraining,
	}, workflows.TrainingPipelineWorkflow, workflows.TrainingPipelineInput{
		JobID:           jobID,
		TaskType:        req.TaskType,
		Backend:         req.Backend,
		UseKatib:        req.UseKatib,
		AutoDeploy:      req.AutoDeploy,
		GradientSteps:   req.GradientSteps,
		MinTransitions:  req.MinTransitions,
		MaxDatasetFiles: req.MaxDatasetFiles,
	})
	if err != nil {
		return nil, fmt.Errorf("start training pipeline for tenant %s: %w", tenantID, err)
	}
	return &PipelineRun{WorkflowID: run.GetID(), JobID: jobID, Status: PipelineStatusRunning}, nil
}

func validatePipeline(req PipelineRequest) error {
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), ErrInvalidInput)
	}
	switch {
	case !pipelineTasks[req.TaskType]:
		return invalid("task_type %q is not supported (locomotion)", req.TaskType)
	case !pipelineBackends[req.Backend]:
		return invalid("backend %q is not supported (argo, kubeflow)", req.Backend)
	case req.GradientSteps < 0 || req.MinTransitions < 0 || req.MaxDatasetFiles < 0:
		return invalid("gradient_steps, min_transitions and max_dataset_files must not be negative")
	case req.UseKatib && req.Backend == "argo":
		return invalid("use_katib requires the kubeflow backend")
	}
	return nil
}

func (s *pipelineService) GetPipeline(ctx context.Context, workflowID string) (*PipelineRun, error) {
	if s.temporal == nil {
		return nil, fmt.Errorf("get pipeline: temporal not configured: %w", ErrUnavailable)
	}
	desc, err := s.temporal.DescribeWorkflowExecution(ctx, workflowID, "")
	if err != nil {
		var notFound *serviceerror.NotFound
		if errors.As(err, &notFound) {
			return nil, fmt.Errorf("pipeline %s: %w", workflowID, ErrNotFound)
		}
		return nil, fmt.Errorf("describe pipeline %s: %w", workflowID, err)
	}

	run := &PipelineRun{WorkflowID: workflowID}
	switch desc.GetWorkflowExecutionInfo().GetStatus() {
	case enums.WORKFLOW_EXECUTION_STATUS_RUNNING, enums.WORKFLOW_EXECUTION_STATUS_CONTINUED_AS_NEW:
		run.Status = PipelineStatusRunning
	case enums.WORKFLOW_EXECUTION_STATUS_COMPLETED:
		run.Status = PipelineStatusCompleted
		var result workflows.TrainingPipelineResult
		if err := s.temporal.GetWorkflow(ctx, workflowID, "").Get(ctx, &result); err != nil {
			return nil, fmt.Errorf("read pipeline %s result: %w", workflowID, err)
		}
		run.JobID = result.JobID
		run.Result = &result
	default:
		run.Status = PipelineStatusFailed
	}
	return run, nil
}
