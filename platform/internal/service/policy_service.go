package service

import (
	"context"
	"errors"
	"fmt"

	"github.com/dimuthu/robot-fleet/internal/store"
)

// Where a robot's policy assignment came from.
const (
	PolicySourceAssigned     = "assigned"      // the robot's own (canary) assignment
	PolicySourceFleetDefault = "fleet_default" // the fleet's deployed model
)

// RobotPolicy tells a robot which low-level policy to execute locally. The
// platform decides the model (rollouts, canaries); the robot only runs it.
type RobotPolicy struct {
	RobotID     string `json:"robot_id"`
	ModelID     string `json:"model_id"`
	Version     string `json:"version"`
	ArtifactURL string `json:"artifact_url"`
	Source      string `json:"source"`
}

// PolicyService resolves which model a robot should execute.
type PolicyService interface {
	GetRobotPolicy(ctx context.Context, robotID string) (*RobotPolicy, error)
}

// RobotGetter reads one robot's record.
type RobotGetter interface {
	GetRobot(ctx context.Context, id string) (*store.RobotRecord, error)
}

// ModelReader reads model registry records.
type ModelReader interface {
	GetModel(ctx context.Context, id string) (*store.ModelRecord, error)
	ListModels(ctx context.Context, status string) ([]*store.ModelRecord, error)
}

type policyService struct {
	robots RobotGetter
	models ModelReader
}

// NewPolicyService creates a PolicyService.
func NewPolicyService(robots RobotGetter, models ModelReader) PolicyService {
	return &policyService{robots: robots, models: models}
}

// GetRobotPolicy returns the robot's assigned model if it still exists,
// otherwise the fleet's deployed model. Robots that haven't registered yet
// get the fleet model too. ErrNotFound means nothing is deployed.
func (s *policyService) GetRobotPolicy(ctx context.Context, robotID string) (*RobotPolicy, error) {
	robot, err := s.robots.GetRobot(ctx, robotID)
	switch {
	case errors.Is(err, store.ErrNotFound):
		robot = &store.RobotRecord{ID: robotID}
	case err != nil:
		return nil, fmt.Errorf("get robot %s: %w", robotID, err)
	}

	if robot.InferenceModelID != "" {
		model, err := s.models.GetModel(ctx, robot.InferenceModelID)
		switch {
		case err == nil:
			return toPolicy(robotID, model, PolicySourceAssigned), nil
		case !errors.Is(err, store.ErrNotFound):
			return nil, fmt.Errorf("get assigned model %s: %w", robot.InferenceModelID, err)
		}
		// Assigned model no longer exists: fall back to the fleet model.
	}

	deployed, err := s.models.ListModels(ctx, ModelStatusDeployed)
	if err != nil {
		return nil, fmt.Errorf("list deployed models: %w", err)
	}
	if len(deployed) == 0 {
		return nil, fmt.Errorf("no policy for robot %s: nothing deployed: %w", robotID, ErrNotFound)
	}
	return toPolicy(robotID, deployed[0], PolicySourceFleetDefault), nil
}

func toPolicy(robotID string, m *store.ModelRecord, source string) *RobotPolicy {
	return &RobotPolicy{RobotID: robotID, ModelID: m.ID, Version: m.Version, ArtifactURL: m.ArtifactURL, Source: source}
}
