package experience

import (
	"strings"
	"testing"
	"time"

	"github.com/dimuthu/robot-fleet/internal/telemetry"
)

func state(tc *telemetry.TrainingContext) *telemetry.RobotState {
	return &telemetry.RobotState{
		RobotId: "robot-0001",
		Pose: &telemetry.Pose{
			Position:    &telemetry.Vector3{X: 1, Y: 2, Z: 1.2},
			Orientation: &telemetry.Quaternion{W: 1},
		},
		Joints: []*telemetry.JointState{
			{Name: "left_knee", Position: 0.1, Velocity: 0.2},
			{Name: "right_knee", Position: 0.3, Velocity: 0.4},
		},
		TrainingContext: tc,
	}
}

func TestBuildTransition_IncludesPolicyInputAndEpisodeKeys(t *testing.T) {
	obs := make([]float64, ObservationDim)
	obs[0] = 1.25
	tc := &telemetry.TrainingContext{
		ActionTaken: make([]float64, ActionDim),
		StepReward:  0.7,
		TaskCommand: "walk",
		EpisodeDone: true,
		Observation: obs,
		EpisodeId:   "ep-1",
		Step:        42,
		Exploration: true,
		Objects: []*telemetry.ObjectState{
			{Name: "red_box", Position: &telemetry.Vector3{X: 1}, Velocity: &telemetry.Vector3{}},
		},
	}
	now := time.Unix(1_800_000_000, 0)

	rec, ok := BuildTransition(state(tc), now)
	if !ok {
		t.Fatal("expected a transition")
	}
	for key, want := range map[string]any{
		"robot_id": "robot-0001", "episode_id": "ep-1", "step": int64(42), "exploration": true,
		"reward": 0.7, "episode_done": true, "task_command": "walk", "ts": now.Unix(),
	} {
		if rec[key] != want {
			t.Errorf("%s = %v (%T), want %v", key, rec[key], rec[key], want)
		}
	}
	if got := rec["obs"].([]float64); len(got) != ObservationDim || got[0] != 1.25 {
		t.Errorf("obs = %v", got)
	}
	if len(rec["action"].([]float64)) != ActionDim {
		t.Errorf("action length = %d", len(rec["action"].([]float64)))
	}
	if jp := rec["joint_pos"].([]float64); len(jp) != 2 || jp[1] != 0.3 {
		t.Errorf("joint_pos = %v", jp)
	}
	if q := rec["root_quat"].([]float64); q[3] != 1 {
		t.Errorf("root_quat must be x,y,z,w; got %v", q)
	}
	if objs := rec["objects"].([]map[string]any); len(objs) != 1 || objs[0]["name"] != "red_box" {
		t.Errorf("objects = %v", objs)
	}
}

func TestBuildTransition_SkipsPacketsWithoutTrainingData(t *testing.T) {
	if _, ok := BuildTransition(state(nil), time.Now()); ok {
		t.Error("no training context → no transition")
	}
	if _, ok := BuildTransition(state(&telemetry.TrainingContext{}), time.Now()); ok {
		t.Error("training context without an action → no transition")
	}
}

func TestBuildTransition_LegacyContextWithoutObservation(t *testing.T) {
	// Older simulators send no observation; keep the record (it is still
	// telemetry) but without an "obs" key so trainers can skip it.
	rec, ok := BuildTransition(state(&telemetry.TrainingContext{ActionTaken: []float64{0.1}}), time.Now())
	if !ok {
		t.Fatal("expected a transition")
	}
	if _, has := rec["obs"]; has {
		t.Error("legacy record must not carry an empty obs")
	}
}

func TestObjectKey(t *testing.T) {
	ts := time.Date(2026, 10, 18, 7, 0, 0, 5, time.UTC)
	key := ObjectKey("robot-0001", ts)
	if key != "training_data/v3/2026/10/18/07/robot-0001/transitions_1792306800000000005.ndjson" {
		t.Errorf("key = %q", key)
	}
	if !strings.HasPrefix(key, Prefix) {
		t.Errorf("key must live under %q", Prefix)
	}
}

func TestPrefixIsVersionedAwayFromLegacyRecords(t *testing.T) {
	// Older records never match: pre-v2 (no observation, directly under
	// training_data/YYYY/...) and v2 (world-frame observation).
	for _, legacy := range []string{
		"training_data/2026/10/18/07/robot-0001/transitions_1.ndjson",
		"training_data/v2/2026/10/18/07/robot-0001/transitions_1.ndjson",
	} {
		if strings.HasPrefix(legacy, Prefix) {
			t.Errorf("legacy key %q must not match Prefix %q", legacy, Prefix)
		}
	}
}
