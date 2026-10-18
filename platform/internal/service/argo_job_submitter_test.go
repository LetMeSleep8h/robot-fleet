package service

import (
	"context"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/store"
)

type recordedSubmit struct {
	template string
	params   map[string]string
	labels   map[string]string
}

type fakeWorkflowSubmitter struct{ calls []recordedSubmit }

func (f *fakeWorkflowSubmitter) SubmitTemplate(_ context.Context, template string, params, labels map[string]string) (string, error) {
	f.calls = append(f.calls, recordedSubmit{template, params, labels})
	return template + "-xyz", nil
}

func TestArgoJobSubmitter_Training(t *testing.T) {
	tests := []struct {
		algorithm    string
		wantTemplate string
		wantEnv      bool
	}{
		{"PPO", "fleetos-locomotion-training", true},
		{"SAC", "fleetos-reach-training", false},
	}
	for _, tt := range tests {
		t.Run(tt.algorithm, func(t *testing.T) {
			wf := &fakeWorkflowSubmitter{}
			s := NewArgoJobSubmitter(wf)
			job := &store.TrainingJobRecord{ID: "job-1", Algorithm: tt.algorithm, Environment: "FleetOS-Humanoid-v2", Timesteps: 20000}

			if err := s.SubmitTrainingJob(context.Background(), job); err != nil {
				t.Fatalf("SubmitTrainingJob: %v", err)
			}
			c := wf.calls[0]
			if c.template != tt.wantTemplate {
				t.Errorf("template = %q, want %q", c.template, tt.wantTemplate)
			}
			if c.params["job-id"] != "job-1" || c.params["timesteps"] != "20000" {
				t.Errorf("params = %v", c.params)
			}
			if _, hasEnv := c.params["env-id"]; hasEnv != tt.wantEnv {
				t.Errorf("env-id present = %v, want %v", hasEnv, tt.wantEnv)
			}
			if c.labels["fleetos.io/job-id"] != "job-1" {
				t.Errorf("labels = %v", c.labels)
			}
		})
	}
}

func TestArgoJobSubmitter_Eval(t *testing.T) {
	wf := &fakeWorkflowSubmitter{}
	s := NewArgoJobSubmitter(wf)
	job := &store.TrainingJobRecord{ID: "job-1", Algorithm: "PPO", Environment: "FleetOS-Humanoid-v2", ArtifactURL: "training/job-1/policy.zip"}
	eval := &store.TrainingEvalRecord{ID: "eval-9", JobID: "job-1", ScenariosTotal: 50}

	if err := s.SubmitEvalJob(context.Background(), eval, job); err != nil {
		t.Fatalf("SubmitEvalJob: %v", err)
	}
	c := wf.calls[0]
	if c.template != "fleetos-locomotion-eval" {
		t.Errorf("template = %q", c.template)
	}
	want := map[string]string{"eval-id": "eval-9", "artifact-url": "training/job-1/policy.zip", "scenarios": "50", "env-id": "FleetOS-Humanoid-v2"}
	for k, v := range want {
		if c.params[k] != v {
			t.Errorf("param %s = %q, want %q", k, c.params[k], v)
		}
	}
}

func TestArgoJobSubmitter_UnsupportedAlgorithm(t *testing.T) {
	s := NewArgoJobSubmitter(&fakeWorkflowSubmitter{})
	if err := s.SubmitTrainingJob(context.Background(), &store.TrainingJobRecord{ID: "j", Algorithm: "DQN"}); err == nil {
		t.Fatal("expected error")
	}
}
