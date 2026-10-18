package activities

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/dimuthu/robot-fleet/internal/store"
)

type fakePromoter struct {
	promoted []string
	err      error
}

func (f *fakePromoter) PromoteModel(_ context.Context, id string) error {
	f.promoted = append(f.promoted, id)
	return f.err
}

type fakeRolloutRobots struct {
	robots    []*store.RobotRecord
	assigned  map[string]string
	assignErr error
}

func (f *fakeRolloutRobots) ListAllActiveRobots(_ context.Context, _ time.Time, _ int) ([]*store.RobotRecord, error) {
	return f.robots, nil
}

func (f *fakeRolloutRobots) UpdateRobotInferenceModel(_ context.Context, robotID, modelID string) error {
	if f.assignErr != nil {
		return f.assignErr
	}
	if f.assigned == nil {
		f.assigned = map[string]string{}
	}
	f.assigned[robotID] = modelID
	return nil
}

func (f *fakeRolloutRobots) ListRobotsByInferenceModel(_ context.Context, modelID string) ([]*store.RobotRecord, error) {
	var out []*store.RobotRecord
	for _, r := range f.robots {
		if f.assigned[r.ID] == modelID {
			out = append(out, r)
		}
	}
	return out, nil
}

func TestFinalizeDeployment_PromotesAndAssignsFleet(t *testing.T) {
	models := newFakeModelRepo(
		&store.ModelRecord{ID: "new", Status: "canary"},
		&store.ModelRecord{ID: "old", Status: "deployed"},
	)
	robots := &fakeRolloutRobots{robots: []*store.RobotRecord{{ID: "r1"}, {ID: "r2"}}}
	promoter := &fakePromoter{}
	acts := &DeploymentActivities{Models: models, Robots: robots, ModelService: promoter}

	if err := acts.FinalizeDeployment(context.Background(), FinalizeInput{ModelID: "new", BaselineModelID: "old"}); err != nil {
		t.Fatalf("FinalizeDeployment: %v", err)
	}
	if len(promoter.promoted) != 1 || promoter.promoted[0] != "new" {
		t.Errorf("model must be promoted (publishes the inference hot-swap event), got %v", promoter.promoted)
	}
	if models.models["old"].Status != "archived" {
		t.Errorf("baseline status = %s, want archived", models.models["old"].Status)
	}
	if robots.assigned["r1"] != "new" || robots.assigned["r2"] != "new" {
		t.Errorf("all robots should run the new model, got %v", robots.assigned)
	}
}

func TestFinalizeDeployment_Errors(t *testing.T) {
	tests := []struct {
		name     string
		promoter *fakePromoter
		robots   *fakeRolloutRobots
	}{
		{"promotion fails", &fakePromoter{err: errors.New("redis down")}, &fakeRolloutRobots{}},
		{"robot assignment fails", &fakePromoter{}, &fakeRolloutRobots{robots: []*store.RobotRecord{{ID: "r1"}}, assignErr: errors.New("pg down")}},
		{"no promoter configured", nil, &fakeRolloutRobots{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acts := &DeploymentActivities{
				Models: newFakeModelRepo(&store.ModelRecord{ID: "new", Status: "canary"}),
				Robots: tt.robots,
			}
			if tt.promoter != nil {
				acts.ModelService = tt.promoter
			}
			if err := acts.FinalizeDeployment(context.Background(), FinalizeInput{ModelID: "new"}); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
