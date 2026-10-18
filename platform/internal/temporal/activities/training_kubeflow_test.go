package activities

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dimuthu/robot-fleet/internal/kube"
	"github.com/dimuthu/robot-fleet/internal/kubeflow"
)

// fakeKubeflow is a scripted KubeflowRunner.
type fakeKubeflow struct {
	mu          sync.Mutex
	trainJobs   []kubeflow.TrainJobSpec
	experiments []kubeflow.ExperimentSpec
	existing    map[string]bool // names that already exist (simulates a retried activity)
	jobStatus   map[string]kubeflow.JobStatus
	expStatus   kubeflow.ExperimentStatus
	expPolls    int
}

func (f *fakeKubeflow) SubmitTrainJob(_ context.Context, spec kubeflow.TrainJobSpec) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.existing[spec.Name] {
		return "", kube.ErrAlreadyExists
	}
	f.trainJobs = append(f.trainJobs, spec)
	return spec.Name, nil
}

func (f *fakeKubeflow) TrainJobStatus(_ context.Context, name string) (kubeflow.JobStatus, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if s, ok := f.jobStatus[name]; ok {
		return s, nil
	}
	return kubeflow.JobStatus{Done: true, Succeeded: true}, nil
}

func (f *fakeKubeflow) SubmitExperiment(_ context.Context, spec kubeflow.ExperimentSpec) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.existing[spec.Name] {
		return "", kube.ErrAlreadyExists
	}
	f.experiments = append(f.experiments, spec)
	return spec.Name, nil
}

func (f *fakeKubeflow) ExperimentStatus(_ context.Context, _ string) (kubeflow.ExperimentStatus, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.expPolls++
	if f.expPolls < 2 {
		return kubeflow.ExperimentStatus{}, nil // still running on the first poll
	}
	return f.expStatus, nil
}

func cmd(spec kubeflow.TrainJobSpec) string { return strings.Join(spec.Command, " ") }

func TestSubmitTrainingRun_KubeflowTrainsThenEvaluates(t *testing.T) {
	kf := &fakeKubeflow{}
	acts := &TrainingActivities{Kubeflow: kf, ModelsBucket: "fleetos-models", PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	val, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{
		JobID: "job-9", TaskType: "locomotion", Timesteps: 20000, EnvID: "FleetOS-Humanoid-v2",
		Backend: BackendKubeflow, Hyperparameters: map[string]string{"learning_rate": "0.0012"},
	})
	if err != nil {
		t.Fatalf("SubmitTrainingRun: %v", err)
	}
	var out SubmitTrainingRunOutput
	_ = val.Get(&out)

	if len(kf.trainJobs) != 2 {
		t.Fatalf("expected train + eval TrainJobs, got %d", len(kf.trainJobs))
	}
	train, eval := kf.trainJobs[0], kf.trainJobs[1]
	for _, want := range []string{"train_locomotion.py", "--job-id=job-9", "--timesteps=20000", "--env-id=FleetOS-Humanoid-v2", "--learning-rate=0.0012"} {
		if !strings.Contains(cmd(train), want) {
			t.Errorf("train command %q missing %q", cmd(train), want)
		}
	}
	for _, want := range []string{"evaluate_policy.py", "--eval-id=job-9", "--model-path=s3://fleetos-models/training/job-9/policy.zip"} {
		if !strings.Contains(cmd(eval), want) {
			t.Errorf("eval command %q missing %q", cmd(eval), want)
		}
	}
	if train.Labels[LabelJobID] != "job-9" {
		t.Errorf("labels = %v", train.Labels)
	}
	if out.ModelPath != "training/job-9/policy.zip" || out.RunID != train.Name {
		t.Errorf("unexpected output %+v", out)
	}
}

func TestSubmitTrainingRun_KubeflowRetryReusesExistingJobs(t *testing.T) {
	kf := &fakeKubeflow{existing: map[string]bool{"train-job-9": true, "eval-job-9": true}}
	acts := &TrainingActivities{Kubeflow: kf, ModelsBucket: "fleetos-models", PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	if _, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{
		JobID: "job-9", TaskType: "locomotion", Backend: BackendKubeflow,
	}); err != nil {
		t.Fatalf("retried run should resume existing TrainJobs: %v", err)
	}
	if len(kf.trainJobs) != 0 {
		t.Errorf("no new TrainJobs expected on retry, got %d", len(kf.trainJobs))
	}
}

func TestSubmitTrainingRun_KubeflowFailures(t *testing.T) {
	tests := []struct {
		name  string
		acts  *TrainingActivities
		input SubmitTrainingRunInput
	}{
		{"not configured", &TrainingActivities{}, SubmitTrainingRunInput{JobID: "j", TaskType: "locomotion", Backend: BackendKubeflow}},
		{"train job fails", &TrainingActivities{Kubeflow: &fakeKubeflow{jobStatus: map[string]kubeflow.JobStatus{"train-j": {Done: true, Message: "OOMKilled"}}}},
			SubmitTrainingRunInput{JobID: "j", TaskType: "locomotion", Backend: BackendKubeflow}},
		{"unknown backend", &TrainingActivities{Kubeflow: &fakeKubeflow{}}, SubmitTrainingRunInput{JobID: "j", TaskType: "locomotion", Backend: "sagemaker"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.acts.PollInterval = time.Millisecond
			env := newActivityEnv(t, tt.acts)
			_, err := env.ExecuteActivity(tt.acts.SubmitTrainingRun, tt.input)
			requireNonRetryable(t, err)
		})
	}
}

func TestSubmitTrainingRun_KubeflowJobNamesAreDNSSafe(t *testing.T) {
	kf := &fakeKubeflow{}
	acts := &TrainingActivities{Kubeflow: kf, ModelsBucket: "b", PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	if _, err := env.ExecuteActivity(acts.SubmitTrainingRun, SubmitTrainingRunInput{
		JobID: "Train_123.ABC", TaskType: "locomotion", Backend: BackendKubeflow,
	}); err != nil {
		t.Fatal(err)
	}
	for _, j := range kf.trainJobs {
		if j.Name != strings.ToLower(j.Name) || strings.ContainsAny(j.Name, "_.") || len(j.Name) > maxKubeNameLen {
			t.Errorf("TrainJob name %q is not a valid DNS-1123 label", j.Name)
		}
	}
}

func TestRunHyperparameterSearch(t *testing.T) {
	kf := &fakeKubeflow{expStatus: kubeflow.ExperimentStatus{
		Done: true, Succeeded: true, BestTrial: "hpo-job-9-x",
		BestParams: map[string]string{"learning_rate": "0.0012", "n_steps": "1024"}, BestObjective: 17.25,
	}}
	acts := &TrainingActivities{
		Kubeflow:     kf,
		HPO:          HPOSettings{Algorithm: "tpe", MaxTrials: 4, ParallelTrials: 2, TrialTimesteps: 20000},
		PollInterval: time.Millisecond,
	}
	env := newActivityEnv(t, acts)

	val, err := env.ExecuteActivity(acts.RunHyperparameterSearch, HyperparameterSearchInput{JobID: "job-9", TaskType: "locomotion", EnvID: "FleetOS-Humanoid-v2"})
	if err != nil {
		t.Fatalf("RunHyperparameterSearch: %v", err)
	}
	var out HyperparameterSearchOutput
	_ = val.Get(&out)

	exp := kf.experiments[0]
	if exp.Name != "hpo-job-9" || exp.Algorithm != "tpe" || exp.MaxTrials != 4 || exp.ParallelTrials != 2 || exp.TrialTimesteps != 20000 {
		t.Errorf("unexpected experiment %+v", exp)
	}
	if out.Experiment != "hpo-job-9" || out.BestObjective != 17.25 || out.BestParams["learning_rate"] != "0.0012" {
		t.Errorf("unexpected output %+v", out)
	}
}

func TestRunHyperparameterSearch_Failures(t *testing.T) {
	tests := []struct {
		name  string
		kf    *fakeKubeflow
		input HyperparameterSearchInput
	}{
		{"not configured", nil, HyperparameterSearchInput{JobID: "j", TaskType: "locomotion"}},
		{"task without tunable flags", &fakeKubeflow{}, HyperparameterSearchInput{JobID: "j", TaskType: "reach"}},
		{"experiment failed", &fakeKubeflow{expStatus: kubeflow.ExperimentStatus{Done: true, Message: "all trials failed"}}, HyperparameterSearchInput{JobID: "j", TaskType: "locomotion"}},
		{"no best trial", &fakeKubeflow{expStatus: kubeflow.ExperimentStatus{Done: true, Succeeded: true}}, HyperparameterSearchInput{JobID: "j", TaskType: "locomotion"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acts := &TrainingActivities{HPO: DefaultHPOSettings(), PollInterval: time.Millisecond}
			if tt.kf != nil {
				acts.Kubeflow = tt.kf
			}
			env := newActivityEnv(t, acts)
			_, err := env.ExecuteActivity(acts.RunHyperparameterSearch, tt.input)
			requireNonRetryable(t, err)
		})
	}
}

func TestKubeNames_FitKatibLimitAndStayUnique(t *testing.T) {
	a := kubeName("hpo", "pipe-575af9db-0b1b-4ef0-9f12-889f8ed0d4f5", maxKatibNameLen)
	b := kubeName("hpo", "pipe-575af9db-0b1b-4ef0-9f12-000000000000", maxKatibNameLen)
	if len(a) > maxKatibNameLen || len(b) > maxKatibNameLen {
		t.Errorf("names exceed Katib's %d-char limit: %q (%d), %q (%d)", maxKatibNameLen, a, len(a), b, len(b))
	}
	if a == b {
		t.Errorf("truncated names collide: %q", a)
	}
	if a != kubeName("hpo", "pipe-575af9db-0b1b-4ef0-9f12-889f8ed0d4f5", maxKatibNameLen) {
		t.Error("names must be deterministic so retries find their objects")
	}
	if short := kubeName("hpo", "job-1", maxKatibNameLen); short != "hpo-job-1" {
		t.Errorf("short names must be unchanged, got %q", short)
	}
}

func TestRunHyperparameterSearch_LongJobIDFitsKatib(t *testing.T) {
	kf := &fakeKubeflow{expStatus: kubeflow.ExperimentStatus{Done: true, Succeeded: true, BestParams: map[string]string{"n_steps": "512"}}}
	acts := &TrainingActivities{Kubeflow: kf, HPO: DefaultHPOSettings(), PollInterval: time.Millisecond}
	env := newActivityEnv(t, acts)

	if _, err := env.ExecuteActivity(acts.RunHyperparameterSearch, HyperparameterSearchInput{
		JobID: "pipe-575af9db-0b1b-4ef0-9f12-889f8ed0d4f5", TaskType: "locomotion",
	}); err != nil {
		t.Fatal(err)
	}
	if n := kf.experiments[0].Name; len(n) > maxKatibNameLen {
		t.Errorf("experiment name %q is %d chars, Katib allows %d", n, len(n), maxKatibNameLen)
	}
}
