package workflows

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/mock"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"

	"github.com/dimuthu/robot-fleet/internal/temporal/activities"
)

const testManifest = "datasets/job-1/manifest.json"

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

func (p *pipelineEnv) datasetReady() *activities.SnapshotDatasetInput {
	var got activities.SnapshotDatasetInput
	p.env.OnActivity(p.acts.SnapshotDataset, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.SnapshotDatasetInput) (*activities.SnapshotDatasetOutput, error) {
			got = in
			return &activities.SnapshotDatasetOutput{ManifestKey: testManifest, Files: 40, EstimatedTransitions: 4000}, nil
		})
	return &got
}

func (p *pipelineEnv) trainingSucceeds() *activities.SubmitTrainingRunInput {
	var got activities.SubmitTrainingRunInput
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.SubmitTrainingRunInput) (*activities.SubmitTrainingRunOutput, error) {
			got = in
			return &activities.SubmitTrainingRunOutput{RunID: "wf-1", Status: "Succeeded", ModelPath: "training/job-1/policy.zip"}, nil
		})
	return &got
}

func (p *pipelineEnv) result(t *testing.T) TrainingPipelineResult {
	t.Helper()
	var r TrainingPipelineResult
	if err := p.env.GetWorkflowResult(&r); err != nil {
		t.Fatalf("get result: %v", err)
	}
	return r
}

func TestTrainingPipeline_TrainsOfflineOnSnapshotAndRecordsLineage(t *testing.T) {
	p := newPipelineEnv(t)
	snap := p.datasetReady()
	train := p.trainingSucceeds()
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
			return "cql-fleet-v1", nil
		})
	p.env.OnWorkflow(ModelDeploymentWorkflow, mock.Anything, mock.Anything).Return(
		&DeploymentWorkflowResult{ModelID: "cql-fleet-v1", Phase: "deployed", Percent: 100}, nil)

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1", GradientSteps: 5000, MaxDatasetFiles: 50, AutoDeploy: true})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	if snap.JobID != "job-1" || snap.MaxFiles != 50 || snap.MinTransitions != DefaultMinTransitions {
		t.Errorf("snapshot input = %+v", *snap)
	}
	if train.DatasetManifest != testManifest || train.GradientSteps != 5000 || train.Backend != "" {
		t.Errorf("training must use the snapshot, got %+v", *train)
	}
	if gotEval.JobID != "job-1" {
		t.Errorf("evaluation must read the job's own results, got %q", gotEval.JobID)
	}
	if gotReg.DatasetManifest != testManifest || gotReg.DatasetFiles != 40 || gotReg.Algorithm != OfflineAlgorithm || gotReg.ArtifactURL != "training/job-1/policy.zip" {
		t.Errorf("registry must record the model's data lineage, got %+v", gotReg)
	}
	r := p.result(t)
	if r.Phase != "deployed" || r.DatasetManifest != testManifest || r.DatasetFiles != 40 {
		t.Errorf("unexpected result %+v", r)
	}
}

func TestTrainingPipeline_InsufficientDataStopsCleanly(t *testing.T) {
	p := newPipelineEnv(t)
	p.env.OnActivity(p.acts.SnapshotDataset, mock.Anything, mock.Anything).Return(nil,
		temporal.NewNonRetryableApplicationError("not enough fleet experience: 3 batches", activities.ErrTypeInsufficientData, nil))
	trained := false
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(
		func(context.Context, activities.SubmitTrainingRunInput) (*activities.SubmitTrainingRunOutput, error) {
			trained = true
			return nil, nil
		})

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1"})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("too little data is an expected outcome, not a failure: %v", err)
	}
	r := p.result(t)
	if r.Phase != "insufficient_data" || r.GateReason == "" {
		t.Errorf("unexpected result %+v", r)
	}
	if trained {
		t.Error("must not train without enough data")
	}
}

func TestTrainingPipeline_GatedModelIsNotRegistered(t *testing.T) {
	p := newPipelineEnv(t)
	p.datasetReady()
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
	if r.Phase != "gated" || r.GateReason == "" {
		t.Errorf("unexpected result %+v", r)
	}
	if registered {
		t.Error("a model that failed the gate must not be registered")
	}
}

func TestTrainingPipeline_KatibSearchThenKubeflowTraining(t *testing.T) {
	p := newPipelineEnv(t)
	p.datasetReady()
	best := map[string]string{"learning_rate": "0.0012", "cql_alpha": "2.5"}
	var gotSearch activities.HyperparameterSearchInput
	p.env.OnActivity(p.acts.RunHyperparameterSearch, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.HyperparameterSearchInput) (*activities.HyperparameterSearchOutput, error) {
			gotSearch = in
			return &activities.HyperparameterSearchOutput{Experiment: "hpo-job-1", BestTrial: "hpo-job-1-x", BestParams: best, BestObjective: 17.25}, nil
		})
	train := p.trainingSucceeds()
	p.env.OnActivity(p.acts.EvaluateTrainedModel, mock.Anything, mock.Anything).Return(
		&activities.EvaluateModelOutput{PassRate: 0.9, MeanReward: 120, PassesGate: true}, nil)
	var gotReg activities.RegisterTrainedModelInput
	p.env.OnActivity(p.acts.RegisterTrainedModel, mock.Anything, mock.Anything).Return(
		func(_ context.Context, in activities.RegisterTrainedModelInput) (string, error) {
			gotReg = in
			return "cql-fleet-v1", nil
		})

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1", UseKatib: true})

	if err := p.env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
	if gotSearch.JobID != "job-1" || gotSearch.DatasetManifest != testManifest {
		t.Errorf("search must run on the same snapshot, got %+v", gotSearch)
	}
	if train.Backend != activities.BackendKubeflow || train.Hyperparameters["cql_alpha"] != "2.5" || train.DatasetManifest != testManifest {
		t.Errorf("final training must run on Kubeflow with Katib's best params, got %+v", *train)
	}
	if gotReg.Backend != activities.BackendKubeflow || gotReg.Hyperparameters["learning_rate"] != "0.0012" || gotReg.HPOExperiment != "hpo-job-1" {
		t.Errorf("registry must record how the model was produced, got %+v", gotReg)
	}
	if r := p.result(t); r.HPOExperiment != "hpo-job-1" || r.BestParams["cql_alpha"] != "2.5" {
		t.Errorf("result should report the search, got %+v", r)
	}
}

func TestTrainingPipeline_TrainingFailureFailsWorkflow(t *testing.T) {
	p := newPipelineEnv(t)
	p.datasetReady()
	p.env.OnActivity(p.acts.SubmitTrainingRun, mock.Anything, mock.Anything).Return(nil, errors.New("training runner not configured"))

	p.env.ExecuteWorkflow(TrainingPipelineWorkflow, TrainingPipelineInput{JobID: "job-1"})

	if p.env.GetWorkflowError() == nil {
		t.Fatal("expected workflow error when training cannot run")
	}
}
