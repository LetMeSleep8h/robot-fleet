package activities

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strings"
	"time"
)

// ArgoSubmitter submits Argo Workflows to a kind cluster using kubectl.
// Since the Temporal worker runs in Docker Compose (not inside the kind cluster),
// we use exec.Command("kubectl") to submit and poll workflows.
type ArgoSubmitter struct {
	KubeContext string // e.g. "kind-fleetos-training"
	Namespace   string // e.g. "argo"
}

// TrainingRunInput is the input for submitting a training workflow.
type TrainingRunInput struct {
	JobID     string
	TaskType  string // "locomotion" or "reach"
	Timesteps int
}

// RunResult is the result of a completed training workflow.
type RunResult struct {
	Status    string // "Succeeded", "Failed", "Running"
	ModelPath string
}

// SubmitTraining submits an Argo workflow using the installed WorkflowTemplate.
func (s *ArgoSubmitter) SubmitTraining(ctx context.Context, input TrainingRunInput) (string, error) {
	templateName := fmt.Sprintf("fleetos-%s-training", input.TaskType)
	workflowName := fmt.Sprintf("fleetos-%s-%s", input.TaskType, input.JobID)

	cmd := exec.CommandContext(ctx, "kubectl", "create", "-n", s.Namespace,
		"--context", s.KubeContext,
		"-f", "-")

	manifest := fmt.Sprintf(`apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: %s
spec:
  workflowTemplateRef:
    name: %s
  arguments:
    parameters:
      - name: job-id
        value: "%s"
      - name: timesteps
        value: "%d"`, workflowName, templateName, input.JobID, input.Timesteps)

	cmd.Stdin = strings.NewReader(manifest)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("submit argo workflow: %s: %w", string(out), err)
	}

	slog.Info("argo workflow submitted", "name", workflowName, "template", templateName)
	return workflowName, nil
}

// GetRunStatus returns the current phase of the workflow.
func (s *ArgoSubmitter) GetRunStatus(ctx context.Context, workflowName string) (string, error) {
	cmd := exec.CommandContext(ctx, "kubectl", "get", "workflow", workflowName,
		"-n", s.Namespace, "--context", s.KubeContext,
		"-o", "jsonpath={.status.phase}")
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("get workflow status: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// WaitForCompletion polls the workflow until it succeeds, fails, or the context is cancelled.
func (s *ArgoSubmitter) WaitForCompletion(ctx context.Context, workflowName string) (*RunResult, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		status, err := s.GetRunStatus(ctx, workflowName)
		if err != nil {
			slog.Warn("failed to get workflow status", "workflow", workflowName, "error", err)
			time.Sleep(30 * time.Second)
			continue
		}

		switch status {
		case "Succeeded":
			return &RunResult{Status: "Succeeded"}, nil
		case "Failed", "Error":
			return &RunResult{Status: status}, fmt.Errorf("workflow %s %s", workflowName, status)
		default:
			slog.Info("workflow running", "name", workflowName, "phase", status)
			time.Sleep(30 * time.Second)
		}
	}
}
