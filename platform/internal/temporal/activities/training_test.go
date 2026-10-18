package activities

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"

	"github.com/dimuthu/robot-fleet/internal/argo"
	"github.com/dimuthu/robot-fleet/internal/modelgate"
	"github.com/dimuthu/robot-fleet/internal/store"
)

// fakeRunner is a scripted WorkflowRunner.
type fakeRunner struct {
	mu          sync.Mutex
	submitted   []submission
	statuses    []argo.Status // returned in order; last one repeats
	statusErrs  int           // first N status calls fail
	statusCalls int
	submitErr   error
}

type submission struct {
	template string
	params   map[string]string
	labels   map[string]string
}

func (f *fakeRunner) SubmitTemplate(_ context.Context, template string, params, labels map[string]string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.submitErr != nil {
		return "", f.submitErr
	}
	f.submitted = append(f.submitted, submission{template, params, labels})
	return template + "-abc12", nil
}

func (f *fakeRunner) GetWorkflowStatus(_ context.Context, _ string) (argo.Status, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.statusCalls++
	if f.statusCalls <= f.statusErrs {
		return argo.Status{}, errors.New("argo unavailable")
	}
	i := f.statusCalls - f.statusErrs - 1
	if i >= len(f.statuses) {
		i = len(f.statuses) - 1
	}
	return f.statuses[i], nil
}

// fakeArtifacts serves objects from a map.
type fakeArtifacts map[string][]byte

func (f fakeArtifacts) Put(_ context.Context, key string, data []byte, _ string) error {
	f[key] = data
	return nil
}

func (f fakeArtifacts) Get(_ context.Context, key string) ([]byte, error) {
	if b, ok := f[key]; ok {
		return b, nil
	}
	return nil, errors.New("NoSuchKey")
}

// fakeModelRepo is an in-memory store.ModelRepository.
type fakeModelRepo struct {
	mu     sync.Mutex
	models map[string]*store.ModelRecord
}

func newFakeModelRepo(models ...*store.ModelRecord) *fakeModelRepo {
	r := &fakeModelRepo{models: map[string]*store.ModelRecord{}}
	for _, m := range models {
		r.models[m.ID] = m
	}
	return r
}

func (r *fakeModelRepo) RegisterModel(_ context.Context, m *store.ModelRecord) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models[m.ID] = m
	return nil
}

func (r *fakeModelRepo) GetModel(_ context.Context, id string) (*store.ModelRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if m, ok := r.models[id]; ok {
		return m, nil
	}
	return nil, errors.New("not found")
}

func (r *fakeModelRepo) ListModels(_ context.Context, status string) ([]*store.ModelRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []*store.ModelRecord
	for _, m := range r.models {
		if status == "" || m.Status == status {
			out = append(out, m)
		}
	}
	return out, nil
}

func (r *fakeModelRepo) UpdateModelStatus(_ context.Context, id, status string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.models[id]
	if !ok {
		return errors.New("not found")
	}
	m.Status = status
	return nil
}

func newActivityEnv(t *testing.T, acts *TrainingActivities) *testsuite.TestActivityEnvironment {
	t.Helper()
	var suite testsuite.WorkflowTestSuite
	env := suite.NewTestActivityEnvironment()
	env.RegisterActivity(acts)
	return env
}

func requireNonRetryable(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected error")
	}
	var appErr *temporal.ApplicationError
	if !errors.As(err, &appErr) || !appErr.NonRetryable() {
		t.Fatalf("expected non-retryable application error, got %v", err)
	}
}

const testManifest = "datasets/job-7/manifest.json"

func TestSubmitTrainingRun_NoRunnerFailsInsteadOfStubbing(t *testing.T) {
	acts := &TrainingActivities{}
	env := newActivityEnv(t, acts)

	_, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{JobID: "j1", DatasetManifest: testManifest})
	requireNonRetryable(t, err)
}

func TestSubmitTrainingRun_RejectsInvalidRequests(t *testing.T) {
	tests := []struct {
		name  string
		input SubmitTrainingRunInput
	}{
		{"no dataset", SubmitTrainingRunInput{JobID: "j1"}},
		{"unknown hyperparameter", SubmitTrainingRunInput{JobID: "j1", DatasetManifest: testManifest, Hyperparameters: map[string]string{"momentum": "0.9"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner := &fakeRunner{statuses: []argo.Status{{Phase: argo.PhaseSucceeded}}}
			acts := &TrainingActivities{Runner: runner}
			env := newActivityEnv(t, acts)

			_, err := env.ExecuteActivity(acts.SubmitTrainingRun, tt.input)
			requireNonRetryable(t, err)
			if len(runner.submitted) != 0 {
				t.Errorf("nothing should be submitted, got %v", runner.submitted)
			}
		})
	}
}

func TestSubmitTrainingRun_SubmitsOfflineTemplateAndWaits(t *testing.T) {
	runner := &fakeRunner{statuses: []argo.Status{{Phase: argo.PhasePending}, {Phase: argo.PhaseRunning}, {Phase: argo.PhaseSucceeded}}}
	acts := &TrainingActivities{Runner: runner, PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	val, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{
		JobID: "job-7", DatasetManifest: testManifest, GradientSteps: 5000,
		Hyperparameters: map[string]string{"learning_rate": "0.0005", "cql_alpha": "2.0"},
	})
	if err != nil {
		t.Fatalf("SubmitTrainingRun: %v", err)
	}
	var out SubmitTrainingRunOutput
	if err := val.Get(&out); err != nil {
		t.Fatal(err)
	}

	if len(runner.submitted) != 1 {
		t.Fatalf("expected 1 submission, got %d", len(runner.submitted))
	}
	sub := runner.submitted[0]
	if sub.template != OfflineTrainingTemplate {
		t.Errorf("template = %q", sub.template)
	}
	wantParams := map[string]string{
		"job-id": "job-7", "dataset-manifest": testManifest, "gradient-steps": "5000",
		"learning-rate": "0.0005", "cql-alpha": "2.0",
	}
	for k, v := range wantParams {
		if sub.params[k] != v {
			t.Errorf("param %s = %q, want %q", k, sub.params[k], v)
		}
	}
	if sub.labels[LabelJobID] != "job-7" {
		t.Errorf("job label = %q", sub.labels[LabelJobID])
	}
	if out.RunID != OfflineTrainingTemplate+"-abc12" || out.Status != argo.PhaseSucceeded || out.ModelPath != "training/job-7/policy.zip" {
		t.Errorf("unexpected output %+v", out)
	}
	if runner.statusCalls != 3 {
		t.Errorf("expected 3 status polls, got %d", runner.statusCalls)
	}
}

func TestSubmitTrainingRun_WorkflowFailure(t *testing.T) {
	runner := &fakeRunner{statuses: []argo.Status{{Phase: argo.PhaseFailed, Message: "OOMKilled"}}}
	acts := &TrainingActivities{Runner: runner, PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	_, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{JobID: "j", DatasetManifest: testManifest})
	requireNonRetryable(t, err)
}

func TestSubmitTrainingRun_ResumesFromHeartbeatWithoutResubmitting(t *testing.T) {
	runner := &fakeRunner{statuses: []argo.Status{{Phase: argo.PhaseSucceeded}}}
	acts := &TrainingActivities{Runner: runner, PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)
	env.SetHeartbeatDetails("fleetos-offline-training-prev1")

	val, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{JobID: "j", DatasetManifest: testManifest})
	if err != nil {
		t.Fatalf("SubmitTrainingRun: %v", err)
	}
	var out SubmitTrainingRunOutput
	_ = val.Get(&out)
	if len(runner.submitted) != 0 {
		t.Errorf("retry must not resubmit, got %d submissions", len(runner.submitted))
	}
	if out.RunID != "fleetos-offline-training-prev1" {
		t.Errorf("RunID = %q", out.RunID)
	}
}

func TestSubmitTrainingRun_TooManyPollErrors(t *testing.T) {
	runner := &fakeRunner{statusErrs: 1000, statuses: []argo.Status{{Phase: argo.PhaseRunning}}}
	acts := &TrainingActivities{Runner: runner, PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	_, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{JobID: "j", DatasetManifest: testManifest})
	if err == nil {
		t.Fatal("expected error after repeated poll failures")
	}
	if runner.statusCalls != MaxConsecutivePollErrors {
		t.Errorf("expected %d polls, got %d", MaxConsecutivePollErrors, runner.statusCalls)
	}
}

func evalJSON(t *testing.T, passRate, meanReward float64) []byte {
	t.Helper()
	b, err := json.Marshal(map[string]any{"pass_rate": passRate, "mean_reward": meanReward, "scenarios_total": 100})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestEvaluateTrainedModel(t *testing.T) {
	deployed := &store.ModelRecord{ID: "ppo-v1", Status: "deployed", Metrics: map[string]any{"eval_mean_reward": 100.0}}

	tests := []struct {
		name     string
		passRate float64
		reward   float64
		models   []*store.ModelRecord
		baseline string
		wantPass bool
		wantBase float64
	}{
		{"first model passes", 0.8, 50, nil, "", true, 0},
		{"zero pass rate gated", 0.0, 45.6, nil, "", false, 0},
		{"regression vs deployed baseline gated", 0.9, 80, []*store.ModelRecord{deployed}, "", false, 100},
		{"improves on deployed baseline", 0.9, 120, []*store.ModelRecord{deployed}, "", true, 100},
		{"explicit baseline id", 0.9, 120, []*store.ModelRecord{deployed}, "ppo-v1", true, 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acts := &TrainingActivities{
				Models:    newFakeModelRepo(tt.models...),
				Artifacts: fakeArtifacts{"evaluations/job-1/eval_results.json": evalJSON(t, tt.passRate, tt.reward)},
				Gate:      modelgate.DefaultPolicy(),
			}
			env := newActivityEnv(t, acts)

			val, err := env.ExecuteActivity(acts.EvaluateTrainedModel, EvaluateModelInput{JobID: "job-1", BaselineModelID: tt.baseline})
			if err != nil {
				t.Fatalf("EvaluateTrainedModel: %v", err)
			}
			var out EvaluateModelOutput
			if err := val.Get(&out); err != nil {
				t.Fatal(err)
			}
			if out.PassesGate != tt.wantPass {
				t.Errorf("PassesGate = %v (reason %q), want %v", out.PassesGate, out.GateReason, tt.wantPass)
			}
			if !out.PassesGate && out.GateReason == "" {
				t.Error("rejected evaluation must explain why")
			}
			if out.PassRate != tt.passRate || out.MeanReward != tt.reward {
				t.Errorf("metrics not read from eval results: %+v", out)
			}
			if out.BaselineReward != tt.wantBase {
				t.Errorf("BaselineReward = %v, want %v", out.BaselineReward, tt.wantBase)
			}
		})
	}
}

func TestEvaluateTrainedModel_MissingResultsIsError(t *testing.T) {
	acts := &TrainingActivities{Models: newFakeModelRepo(), Artifacts: fakeArtifacts{}, Gate: modelgate.DefaultPolicy()}
	env := newActivityEnv(t, acts)

	if _, err := env.ExecuteActivity(acts.EvaluateTrainedModel, EvaluateModelInput{JobID: "missing"}); err == nil {
		t.Fatal("expected error when eval results are missing")
	}
}

func TestRegisterTrainedModel_RecordsEvalMetrics(t *testing.T) {
	repo := newFakeModelRepo()
	acts := &TrainingActivities{Models: repo}
	env := newActivityEnv(t, acts)

	val, err := env.ExecuteActivity(acts.RegisterTrainedModel, RegisterTrainedModelInput{
		Name: "ppo-lab", Version: "v1", ArtifactURL: "training/j/policy.zip", MeanReward: 120, PassRate: 0.9,
	})
	if err != nil {
		t.Fatalf("RegisterTrainedModel: %v", err)
	}
	var id string
	_ = val.Get(&id)
	m := repo.models[id]
	if m == nil || m.Status != "staged" {
		t.Fatalf("model not registered as staged: %+v", m)
	}
	if err := modelgate.DefaultPolicy().Check(m.Metrics, nil); err != nil {
		t.Errorf("registered metrics should satisfy the gate used at deploy time: %v", err)
	}
}
