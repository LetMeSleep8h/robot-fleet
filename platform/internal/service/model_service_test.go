package service

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/dimuthu/robot-fleet/internal/modelgate"
	"github.com/dimuthu/robot-fleet/internal/store"
)

// mockModelRepo implements store.ModelRepository for testing.
type mockModelRepo struct {
	models map[string]*store.ModelRecord
	err    error
}

func newMockModelRepo() *mockModelRepo {
	return &mockModelRepo{models: make(map[string]*store.ModelRecord)}
}

func (m *mockModelRepo) RegisterModel(_ context.Context, rec *store.ModelRecord) error {
	if m.err != nil {
		return m.err
	}
	m.models[rec.ID] = rec
	return nil
}

func (m *mockModelRepo) GetModel(_ context.Context, id string) (*store.ModelRecord, error) {
	if m.err != nil {
		return nil, m.err
	}
	rec, ok := m.models[id]
	if !ok {
		return nil, ErrNotFound
	}
	return rec, nil
}

func (m *mockModelRepo) ListModels(_ context.Context, status string) ([]*store.ModelRecord, error) {
	if m.err != nil {
		return nil, m.err
	}
	var out []*store.ModelRecord
	for _, r := range m.models {
		if status == "" || r.Status == status {
			out = append(out, r)
		}
	}
	return out, nil
}

func (m *mockModelRepo) UpdateModelStatus(_ context.Context, id, status string) error {
	if m.err != nil {
		return m.err
	}
	rec, ok := m.models[id]
	if !ok {
		return ErrNotFound
	}
	rec.Status = status
	return nil
}

// --- Tests ---

func TestRegisterModel(t *testing.T) {
	repo := newMockModelRepo()
	svc := NewModelRegistryService(repo, newMockCache())

	req := RegisterModelRequest{
		Name:        "groot-n1",
		Version:     "v2.0",
		ArtifactURL: "s3://bucket/groot-n1-v2.0.pt",
		Metrics:     map[string]any{"accuracy": 0.95},
	}
	model, err := svc.RegisterModel(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Status != ModelStatusStaged {
		t.Errorf("expected status=%s, got %s", ModelStatusStaged, model.Status)
	}
	if model.ID != "groot-n1-v2.0" {
		t.Errorf("expected id=groot-n1-v2.0, got %s", model.ID)
	}
}

// fakeCanaryStarter records rollout requests.
type fakeCanaryStarter struct {
	started []string
	err     error
}

func (f *fakeCanaryStarter) StartCanary(_ context.Context, modelID string) (string, error) {
	if f.err != nil {
		return "", f.err
	}
	f.started = append(f.started, modelID)
	return "deploy-" + modelID, nil
}

func evaluatedModel(id string, passRate, reward float64) *store.ModelRecord {
	return &store.ModelRecord{
		ID: id, Name: "ppo", Version: id, ArtifactURL: "training/" + id + "/policy.zip",
		Status:    ModelStatusStaged,
		Metrics:   map[string]any{"eval_pass_rate": passRate, "eval_mean_reward": reward},
		CreatedAt: time.Now(),
	}
}

func TestDeployModel_StartsCanaryWhenGatePasses(t *testing.T) {
	repo := newMockModelRepo()
	repo.models["m1"] = evaluatedModel("m1", 0.9, 120)
	rollout := &fakeCanaryStarter{}
	svc := NewModelRegistryService(repo, newMockCache(), WithCanaryStarter(rollout))

	res, err := svc.DeployModel(context.Background(), "m1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(rollout.started) != 1 || rollout.started[0] != "m1" {
		t.Fatalf("expected canary for m1, got %v", rollout.started)
	}
	if res.WorkflowID != "deploy-m1" || res.Status != DeployStatusCanaryStarted {
		t.Errorf("unexpected result %+v", res)
	}
	// The canary workflow owns status transitions; deploying must not jump straight to deployed.
	if repo.models["m1"].Status != ModelStatusStaged {
		t.Errorf("status changed to %s before the canary ran", repo.models["m1"].Status)
	}
}

func TestDeployModel_Rejections(t *testing.T) {
	deployedBaseline := evaluatedModel("base", 0.95, 100)
	deployedBaseline.Status = ModelStatusDeployed

	tests := []struct {
		name    string
		model   *store.ModelRecord
		extra   []*store.ModelRecord
		rollout *fakeCanaryStarter
		wantErr error
	}{
		{"never evaluated", &store.ModelRecord{ID: "m", Status: ModelStatusStaged}, nil, &fakeCanaryStarter{}, modelgate.ErrRejected},
		{"failed every scenario", evaluatedModel("m", 0, 45.6), nil, &fakeCanaryStarter{}, modelgate.ErrRejected},
		{"regresses vs deployed", evaluatedModel("m", 0.9, 50), []*store.ModelRecord{deployedBaseline}, &fakeCanaryStarter{}, modelgate.ErrRejected},
		{"already deployed", func() *store.ModelRecord {
			m := evaluatedModel("m", 0.9, 120)
			m.Status = ModelStatusDeployed
			return m
		}(), nil, &fakeCanaryStarter{}, ErrConflict},
		{"no rollout configured", evaluatedModel("m", 0.9, 120), nil, nil, ErrUnavailable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := newMockModelRepo()
			repo.models["m"] = tt.model
			for _, m := range tt.extra {
				repo.models[m.ID] = m
			}
			var opts []ModelServiceOption
			if tt.rollout != nil {
				opts = append(opts, WithCanaryStarter(tt.rollout))
			}
			svc := NewModelRegistryService(repo, newMockCache(), opts...)

			_, err := svc.DeployModel(context.Background(), "m")
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			if tt.rollout != nil && len(tt.rollout.started) != 0 {
				t.Errorf("rejected model must not start a canary, got %v", tt.rollout.started)
			}
		})
	}
}

func TestDeployModel_NotFoundError(t *testing.T) {
	repo := newMockModelRepo()
	svc := NewModelRegistryService(repo, newMockCache(), WithCanaryStarter(&fakeCanaryStarter{}))

	if _, err := svc.DeployModel(context.Background(), "nonexistent"); !errors.Is(err, ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestPromoteModel_PublishesEvent(t *testing.T) {
	repo := newMockModelRepo()
	cache := newMockCache()
	svc := NewModelRegistryService(repo, cache)
	repo.models["m1"] = evaluatedModel("m1", 0.9, 120)
	repo.models["m1"].Status = ModelStatusCanary

	if err := svc.PromoteModel(context.Background(), "m1"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo.models["m1"].Status != ModelStatusDeployed {
		t.Errorf("expected model status=deployed, got %s", repo.models["m1"].Status)
	}
	if len(cache.published) != 1 || cache.published[0] != "model:deployed" {
		t.Errorf("expected published channel=model:deployed, got %v", cache.published)
	}
}

func TestPromoteModel_EventPayload(t *testing.T) {
	repo := newMockModelRepo()
	repo.models["m2"] = &store.ModelRecord{
		ID:          "m2",
		Name:        "policy-v3",
		Version:     "v3.1",
		ArtifactURL: "s3://bucket/policy-v3.1.pt",
		Status:      ModelStatusCanary,
		CreatedAt:   time.Now(),
	}
	rc := &recordingCache{}
	svc := NewModelRegistryService(repo, rc)

	if err := svc.PromoteModel(context.Background(), "m2"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var payload map[string]string
	if err := json.Unmarshal(rc.lastPayload, &payload); err != nil {
		t.Fatalf("invalid JSON payload: %v", err)
	}
	if payload["model_id"] != "m2" || payload["version"] != "v3.1" || payload["artifact_url"] != "s3://bucket/policy-v3.1.pt" {
		t.Errorf("unexpected payload %v", payload)
	}
}

func TestArchiveModel(t *testing.T) {
	repo := newMockModelRepo()
	svc := NewModelRegistryService(repo, newMockCache())

	repo.models["m3"] = &store.ModelRecord{ID: "m3", Status: ModelStatusDeployed}

	if err := svc.ArchiveModel(context.Background(), "m3"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo.models["m3"].Status != ModelStatusArchived {
		t.Errorf("expected archived, got %s", repo.models["m3"].Status)
	}
}

// recordingCache captures the last PublishEvent call payload.
type recordingCache struct {
	mockCache
	lastPayload []byte
}

func (r *recordingCache) PublishEvent(_ context.Context, _ string, data []byte) error {
	r.lastPayload = data
	return nil
}
