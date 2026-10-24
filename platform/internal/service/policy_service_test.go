package service

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/store"
)

type fakeRobotGetter map[string]*store.RobotRecord

func (f fakeRobotGetter) GetRobot(_ context.Context, id string) (*store.RobotRecord, error) {
	if r, ok := f[id]; ok {
		return r, nil
	}
	return nil, fmt.Errorf("get robot %s: %w", id, store.ErrNotFound)
}

func TestGetRobotPolicy(t *testing.T) {
	deployed := &store.ModelRecord{ID: "fleet-v3", Version: "v3", ArtifactURL: "training/fleet/policy.zip", Status: ModelStatusDeployed}
	canary := &store.ModelRecord{ID: "canary-v4", Version: "v4", ArtifactURL: "training/canary/policy.zip", Status: ModelStatusCanary}

	tests := []struct {
		name       string
		robots     fakeRobotGetter
		models     []*store.ModelRecord
		robotID    string
		wantModel  string
		wantSource string
		wantErr    error
	}{
		{"canary assignment wins", fakeRobotGetter{"r1": {ID: "r1", InferenceModelID: "canary-v4"}},
			[]*store.ModelRecord{deployed, canary}, "r1", "canary-v4", PolicySourceAssigned, nil},
		{"unassigned robot gets the fleet model", fakeRobotGetter{"r1": {ID: "r1"}},
			[]*store.ModelRecord{deployed, canary}, "r1", "fleet-v3", PolicySourceFleetDefault, nil},
		{"unregistered robot gets the fleet model", fakeRobotGetter{},
			[]*store.ModelRecord{deployed}, "new-robot", "fleet-v3", PolicySourceFleetDefault, nil},
		{"assigned model gone falls back to fleet model", fakeRobotGetter{"r1": {ID: "r1", InferenceModelID: "deleted"}},
			[]*store.ModelRecord{deployed}, "r1", "fleet-v3", PolicySourceFleetDefault, nil},
		{"nothing deployed", fakeRobotGetter{"r1": {ID: "r1"}},
			[]*store.ModelRecord{canary}, "r1", "", "", ErrNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := newMockModelRepo()
			for _, m := range tt.models {
				repo.models[m.ID] = m
			}
			svc := NewPolicyService(tt.robots, repo)

			got, err := svc.GetRobotPolicy(context.Background(), tt.robotID)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("err = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("GetRobotPolicy: %v", err)
			}
			if got.ModelID != tt.wantModel || got.Source != tt.wantSource || got.RobotID != tt.robotID {
				t.Errorf("got %+v, want model %s source %s", got, tt.wantModel, tt.wantSource)
			}
			if got.ArtifactURL == "" || got.Version == "" {
				t.Errorf("policy must carry the artifact and version: %+v", got)
			}
		})
	}
}

func TestGetRobotPolicy_RepositoryErrorsPropagate(t *testing.T) {
	repo := newMockModelRepo()
	repo.err = errors.New("postgres down")
	svc := NewPolicyService(fakeRobotGetter{"r1": {ID: "r1"}}, repo)

	_, err := svc.GetRobotPolicy(context.Background(), "r1")
	if err == nil || errors.Is(err, ErrNotFound) {
		t.Fatalf("a storage failure must not look like 'no policy': %v", err)
	}
}
