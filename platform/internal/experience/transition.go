// Package experience defines the fleet experience records the processor
// writes to S3 and offline training reads back (training_data/ prefix).
package experience

import (
	"fmt"
	"time"

	"github.com/dimuthu/robot-fleet/internal/telemetry"
)

// Policy dimensions shared by the simulator, inference and offline training.
const (
	ObservationDim = 86
	ActionDim      = 25
)

// Prefix is where transition batches live in the telemetry bucket.
// Records gained the full policy observation in format v2; older records
// (directly under training_data/) are never used for training.
const Prefix = "training_data/v2/"

// TransitionsPerBatch is how many transitions the processor buffers per robot
// before flushing a batch file (partial batches are flushed periodically).
const TransitionsPerBatch = 100

// BuildTransition converts a telemetry packet's training context into one
// NDJSON transition record. It returns false when the packet carries no
// training data. Records from simulators that don't send the observation are
// kept without an "obs" key so offline trainers can skip them.
func BuildTransition(state *telemetry.RobotState, now time.Time) (map[string]any, bool) {
	tc := state.GetTrainingContext()
	if tc == nil || len(tc.GetActionTaken()) == 0 {
		return nil, false
	}

	objects := make([]map[string]any, 0, len(tc.GetObjects()))
	for _, obj := range tc.GetObjects() {
		objects = append(objects, map[string]any{
			"name":  obj.GetName(),
			"pos_x": obj.GetPosition().GetX(), "pos_y": obj.GetPosition().GetY(), "pos_z": obj.GetPosition().GetZ(),
			"vel_x": obj.GetVelocity().GetX(), "vel_y": obj.GetVelocity().GetY(), "vel_z": obj.GetVelocity().GetZ(),
		})
	}
	jointPos := make([]float64, 0, len(state.GetJoints()))
	jointVel := make([]float64, 0, len(state.GetJoints()))
	for _, j := range state.GetJoints() {
		jointPos = append(jointPos, j.GetPosition())
		jointVel = append(jointVel, j.GetVelocity())
	}
	pos, rot := state.GetPose().GetPosition(), state.GetPose().GetOrientation()

	rec := map[string]any{
		"robot_id":     state.GetRobotId(),
		"episode_id":   tc.GetEpisodeId(),
		"step":         tc.GetStep(),
		"exploration":  tc.GetExploration(),
		"action":       tc.GetActionTaken(),
		"reward":       tc.GetStepReward(),
		"episode_done": tc.GetEpisodeDone(),
		"task_command": tc.GetTaskCommand(),
		"joint_pos":    jointPos,
		"joint_vel":    jointVel,
		"root_pos":     []float64{pos.GetX(), pos.GetY(), pos.GetZ()},
		"root_quat":    []float64{rot.GetX(), rot.GetY(), rot.GetZ(), rot.GetW()},
		"objects":      objects,
		"ts":           now.Unix(),
	}
	if obs := tc.GetObservation(); len(obs) > 0 {
		rec["obs"] = obs
	}
	return rec, true
}

// ObjectKey is the S3 key for a robot's transition batch flushed at ts.
func ObjectKey(robotID string, ts time.Time) string {
	return fmt.Sprintf("%s%s/%s/transitions_%d.ndjson", Prefix, ts.UTC().Format("2006/01/02/15"), robotID, ts.UnixNano())
}
