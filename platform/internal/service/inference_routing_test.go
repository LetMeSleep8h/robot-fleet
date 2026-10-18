package service

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/command"
	"github.com/dimuthu/robot-fleet/internal/store"
)

type fakeModelLookup map[string]*store.ModelRecord

func (f fakeModelLookup) GetModel(_ context.Context, id string) (*store.ModelRecord, error) {
	if m, ok := f[id]; ok {
		return m, nil
	}
	return nil, ErrNotFound
}

// captureInference records the JSON body sent to /predict.
func captureInference(t *testing.T) (*httptest.Server, *map[string]any) {
	t.Helper()
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode: %v", err)
		}
		_, _ = w.Write([]byte(`{"predicted_actions":[],"model_id":"x"}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

func TestRunInference_RoutesToRobotsAssignedModel(t *testing.T) {
	srv, got := captureInference(t)
	repo := &mockRepo{robots: []*store.RobotRecord{{ID: "r1", InferenceModelID: "ppo-canary-v2"}}}
	models := fakeModelLookup{"ppo-canary-v2": {ID: "ppo-canary-v2", Version: "v2", ArtifactURL: "training/job-2/policy.zip"}}
	svc := NewRobotService(repo, newMockCache(), command.DefaultRegistry(), strings.TrimPrefix(srv.URL, "http://"), 0,
		WithModelLookup(models))

	if _, err := svc.RunInference(context.Background(), InferenceRequest{RobotID: "r1", Instruction: "walk forward"}, "tenant-dev"); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	body := *got
	if body["model_id"] != "ppo-canary-v2" || body["artifact_url"] != "training/job-2/policy.zip" || body["model_version"] != "v2" {
		t.Errorf("inference request must name the robot's model and its artifact, got %v", body)
	}
}

func TestRunInference_UnassignedRobotUsesActiveModel(t *testing.T) {
	srv, got := captureInference(t)
	repo := &mockRepo{robots: []*store.RobotRecord{{ID: "r1"}}}
	svc := NewRobotService(repo, newMockCache(), command.DefaultRegistry(), strings.TrimPrefix(srv.URL, "http://"), 0,
		WithModelLookup(fakeModelLookup{}))

	if _, err := svc.RunInference(context.Background(), InferenceRequest{RobotID: "r1", Instruction: "walk"}, "t"); err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if a, ok := (*got)["artifact_url"]; ok && a != "" {
		t.Errorf("no artifact should be pinned for an unassigned robot, got %v", a)
	}
}
