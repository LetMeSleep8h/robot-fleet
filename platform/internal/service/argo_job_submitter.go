package service

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/dimuthu/robot-fleet/internal/store"
)

// WorkflowSubmitter submits a workflow from a WorkflowTemplate (implemented by *argo.Client).
type WorkflowSubmitter interface {
	SubmitTemplate(ctx context.Context, template string, params, labels map[string]string) (string, error)
}

// argoTemplates names the WorkflowTemplates for one algorithm family.
type argoTemplates struct {
	train    string
	eval     string
	envParam bool // template accepts env-id
}

// templatesByAlgorithm maps a training algorithm to its Argo WorkflowTemplates
// (installed from deploy/k8s/workflows/).
var templatesByAlgorithm = map[string]argoTemplates{
	"PPO": {train: "fleetos-locomotion-training", eval: "fleetos-locomotion-eval", envParam: true},
	"SAC": {train: "fleetos-reach-training", eval: "fleetos-reach-eval"},
}

// labelJobID tags submitted workflows with their FleetOS training job.
const labelJobID = "fleetos.io/job-id"

// ArgoJobSubmitter submits training and evaluation jobs as Argo workflows.
type ArgoJobSubmitter struct {
	workflows WorkflowSubmitter
}

// NewArgoJobSubmitter creates a JobSubmitter backed by Argo Workflows.
func NewArgoJobSubmitter(w WorkflowSubmitter) *ArgoJobSubmitter {
	return &ArgoJobSubmitter{workflows: w}
}

// SubmitTrainingJob starts the algorithm's train+evaluate workflow. The
// workflow's evaluate step reports under the job's ID.
func (a *ArgoJobSubmitter) SubmitTrainingJob(ctx context.Context, job *store.TrainingJobRecord) error {
	tpl, err := templatesFor(job.Algorithm)
	if err != nil {
		return err
	}
	params := map[string]string{
		"job-id":    job.ID,
		"timesteps": strconv.FormatInt(job.Timesteps, 10),
	}
	if tpl.envParam && job.Environment != "" {
		params["env-id"] = job.Environment
	}
	if _, err := a.workflows.SubmitTemplate(ctx, tpl.train, params, map[string]string{labelJobID: job.ID}); err != nil {
		return fmt.Errorf("submit training job %s: %w", job.ID, err)
	}
	return nil
}

// SubmitEvalJob starts an evaluation-only workflow for a completed job.
func (a *ArgoJobSubmitter) SubmitEvalJob(ctx context.Context, eval *store.TrainingEvalRecord, job *store.TrainingJobRecord) error {
	tpl, err := templatesFor(job.Algorithm)
	if err != nil {
		return err
	}
	params := map[string]string{
		"eval-id":      eval.ID,
		"artifact-url": job.ArtifactURL,
		"scenarios":    strconv.Itoa(eval.ScenariosTotal),
	}
	if tpl.envParam && job.Environment != "" {
		params["env-id"] = job.Environment
	}
	if _, err := a.workflows.SubmitTemplate(ctx, tpl.eval, params, map[string]string{labelJobID: job.ID}); err != nil {
		return fmt.Errorf("submit eval %s: %w", eval.ID, err)
	}
	return nil
}

func templatesFor(algorithm string) (argoTemplates, error) {
	tpl, ok := templatesByAlgorithm[strings.ToUpper(algorithm)]
	if !ok {
		return argoTemplates{}, fmt.Errorf("algorithm %q has no training template (supported: PPO, SAC): %w", algorithm, ErrInvalidInput)
	}
	return tpl, nil
}
