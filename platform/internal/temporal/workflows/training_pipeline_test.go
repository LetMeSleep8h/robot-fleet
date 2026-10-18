package workflows

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/mock"
	"go.temporal.io/sdk/testsuite"

	"github.com/dimuthu/robot-fleet/internal/temporal/activities"
)

type pipelineEnv struct {
	env  *testsuite.TestWorkflowEnvironment
	acts *activities.TrainingActivities
}

func newPipelineEnv(t *testing.T) *pipelineEnv {
	t.Helper()
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()
	acts := &activities.TrainingActivities{}
	env.RegisterActivity(acts)
	env.RegisterWorkflow(ModelDeploymentWorkflow)
	env.OnActivity(acts.NotifyTrainingComplete, mock.Anything, mock.Anything, mock.Anything).Return(nil)
	return &pipelineEnv{env: env, acts: acts}
}

func (p *pipelineEnv) trainingSucceeds() {
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(
		&activities.SubmitTrainingRunOutput{RunID: "wf-1", Status: "Succeeded", ModelPath: "training/job-1/policy.zip"}, nil)
}

func (p *pipelineEnv) result(t *testing.T) TrainingPipelineResult {
	t.Helper()
	var r TrainingPipelineResult
	if err := p.env.GetWorkflowResult(&r); err != nil {
		t.Fatalf("get result: %v", err)
	}
	return r
}

func TestTrainingPipeline_GatedModelIsNotRegistered(t *testing.T) {
	p := newPipelineEnv(t)
	p.trainingSucceeds()
	p.env.OnActivity(p.acts.EvaluateTrainedModel, mock.Anything, mock.Anything).Return(
		&activities.EvaluateModelOutput{PassRate: 0, MeanReward: 45.6, PassesGate: false, GateReason: "eval_pass_rate 0.00 is below the minimum 0.50"}, nil)
	registered := false
	p.env.OnActivity(p.acts.RegisterTrainedModel, mock.Anything, mock.Anything).Return(
		func(context.Context, activities.RegisterTrainedModelInput) (string, error) {
			registered = true
			return "x", nil
		})

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1", AutoDeploy: true})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	r := p.result(t)
	if r.Phase != "gated" {
		t.Errorf("phase = %q, want gated", r.Phase)
	}
	if r.GateReason == "" {
		t.Error("gated result should carry the gate reason")
	}
	if registered {
		t.Error("a model that failed the gate must not be registered")
	}
}

func TestTrainingPipeline_PassingModelIsRegisteredWithEvalMetricsAndDeployed(t *testing.T) {
	p := newPipelineEnv(t)
	p.trainingSucceeds()
	var gotEval activities.EvaluateModelInput
	p.env.OnActivity(p.acts.EvaluateTrainedModel, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.EvaluateModelInput) (*activities.EvaluateModelOutput, error) {
			gotEval = in
			return &activities.EvaluateModelOutput{PassRate: 0.9, MeanReward: 120, PassesGate: true}, nil
		})
	var gotReg activities.RegisterTrainedModelInput
	p.env.OnActivity(p.acts.RegisterTrainedModel, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.RegisterTrainedModelInput) (string, error) {
			gotReg = in
			return "ppo-lab-v1", nil
		})
	p.env.OnWorkflow(ModelDeploymentWorkflow, mock.Anything, mock.Anything).Return(
		&DeploymentWorkflowResult{ModelID: "ppo-lab-v1", Phase: "deployed", Percent: 100}, nil)

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1", AutoDeploy: true})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	if gotEval.JobID != "job-1" {
		t.Errorf("evaluation must read the job's own results, got job %q", gotEval.JobID)
	}
	if gotReg.PassRate != 0.9 || gotReg.MeanReward != 120 || gotReg.ArtifactURL != "training/job-1/policy.zip" {
		t.Errorf("unexpected registration %+v", gotReg)
	}
	if r := p.result(t); r.Phase != "deployed" || r.ModelID != "ppo-lab-v1" {
		t.Errorf("unexpected result %+v", r)
	}
}

func TestTrainingPipeline_TrainingFailureFailsWorkflow(t *testing.T) {
	p := newPipelineEnv(t)
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(nil, errors.New("training runner not configured"))

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1"})

	if p.env.GetWorkflowError() == nil {
		t.Fatal("expected workflow error when training cannot run")
	}
}

func TestTrainingPipeline_KatibSearchThenKubeflowTraining(t *testing.T) {
	p := newPipelineEnv(t)
	best := map[string]string{"learning_rate": "0.0012", "n_steps": "1024"}
	var gotSearch activities.HyperparameterSearchInput
	p.env.OnActivity(p.acts.RunHyperparameterSearch, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.HyperparameterSearchInput) (*activities.HyperparameterSearchOutput, error) {
			gotSearch = in
			return &activities.HyperparameterSearchOutput{Experiment: "hpo-job-1", BestTrial: "hpo-job-1-x", BestParams: best, BestObjective: 17.25}, nil
		})
	var gotTrain activities.SubmitTrainingRunInput
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.SubmitTrainingRunInput) (*activities.SubmitTrainingRunOutput, error) {
			gotTrain = in
			return &activities.SubmitTrainingRunOutput{RunID: "train-job-1", Status: "Succeeded", ModelPath: "training/job-1/policy.zip"}, nil
		})
	p.env.OnActivity(p.acts.EvaluateTrainedModel, mock.Anything, mock.Anything).Return(
		&activities.EvaluateModelOutput{PassRate: 0.9, MeanReward: 120, PassesGate: true}, nil)
	var gotReg activities.RegisterTrainedModelInput
	p.env.OnActivity(p.acts.RegisterTrainedModel, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.RegisterTrainedModelInput) (string, error) {
			gotReg = in
			return "ppo-lab-v1", nil
		})

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1", UseKatib: true})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	if gotSearch.JobID != "job-1" || gotSearch.TaskType != "locomotion" {
		t.Errorf("search input = %+v", gotSearch)
	}
	if gotTrain.Backend != activities.BackendKubeflow || gotTrain.Hyperparameters["learning_rate"] != "0.0012" {
		t.Errorf("final training must run on Kubeflow with Katib's best params, got %+v", gotTrain)
	}
	if gotReg.Backend != activities.BackendKubeflow || gotReg.Hyperparameters["n_steps"] != "1024" || gotReg.HPOExperiment != "hpo-job-1" {
		t.Errorf("registry must record how the model was produced, got %+v", gotReg)
	}
	r := p.result(t)
	if r.HPOExperiment != "hpo-job-1" || r.BestParams["learning_rate"] != "0.0012" {
		t.Errorf("result should report the search, got %+v", r)
	}
}

func TestTrainingPipeline_KubeflowBackendPassesThrough(t *testing.T) {
	p := newPipelineEnv(t)
	var gotTrain activities.SubmitTrainingRunInput
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.SubmitTrainingRunInput) (*activities.SubmitTrainingRunOutput, error) {
			gotTrain = in
			return &activities.SubmitTrainingRunOutput{ModelPath: "training/job-1/policy.zip"}, nil
		})
	p.env.OnActivity(p.acts.EvaluateTrainedModel, mock.Anything, mock.Anything).Return(
		&activities.EvaluateModelOutput{PassesGate: false, GateReason: "below"}, nil)

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1", Backend: activities.BackendKubeflow})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	if gotTrain.Backend != activities.BackendKubeflow || len(gotTrain.Hyperparameters) != 0 {
		t.Errorf("unexpected training input %+v", gotTrain)
	}
}
