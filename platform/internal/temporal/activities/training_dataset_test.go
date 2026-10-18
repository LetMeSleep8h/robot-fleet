package activities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/experience"
)

// fakeDataset lists keys under the training_data/ prefix.
type fakeDataset struct {
	keys []string
	err  error
}

func (f fakeDataset) List(_ context.Context, prefix string, _ int) ([]string, error) {
	if f.err != nil {
		return nil, f.err
	}
	var out []string
	for _, k := range f.keys {
		if len(k) >= len(prefix) && k[:len(prefix)] == prefix {
			out = append(out, k)
		}
	}
	return out, nil
}

// memArtifacts is a writable in-memory artifact store.
type memArtifacts map[string][]byte

func (m memArtifacts) Get(_ context.Context, key string) ([]byte, error) {
	if b, ok := m[key]; ok {
		return b, nil
	}
	return nil, errors.New("NoSuchKey")
}

func (m memArtifacts) Put(_ context.Context, key string, data []byte, _ string) error {
	m[key] = data
	return nil
}

func batchKeys(n int) []string {
	keys := make([]string, n)
	for i := range n {
		keys[i] = fmt.Sprintf("%s2026/10/18/07/robot-0001/transitions_%03d.ndjson", experience.Prefix, i)
	}
	return keys
}

func TestSnapshotDataset_WritesManifestOfNewestBatches(t *testing.T) {
	store := memArtifacts{}
	acts := &TrainingActivities{
		Dataset:       fakeDataset{keys: append(batchKeys(30), "lidar/x.bin")},
		DatasetBucket: "fleetos-telemetry",
		Artifacts:     store,
	}
	env := newActivityEnv(t, acts)

	val, err := env.ExecuteActivity(acts.SnapshotDataset, SnapshotDatasetInput{JobID: "job-1", MaxFiles: 20, MinTransitions: 1000})
	if err != nil {
		t.Fatalf("SnapshotDataset: %v", err)
	}
	var out SnapshotDatasetOutput
	_ = val.Get(&out)

	if out.ManifestKey != "datasets/job-1/manifest.json" || out.Files != 20 || out.EstimatedTransitions != 20*experience.TransitionsPerBatch {
		t.Errorf("unexpected output %+v", out)
	}
	var m DatasetManifest
	if err := json.Unmarshal(store[out.ManifestKey], &m); err != nil {
		t.Fatalf("manifest not written as JSON: %v", err)
	}
	if m.JobID != "job-1" || m.Bucket != "fleetos-telemetry" || m.Prefix != experience.Prefix || len(m.Keys) != 20 || m.CreatedAt == "" {
		t.Errorf("unexpected manifest %+v", m)
	}
	// Keys are time-partitioned, so the newest batches sort last.
	if m.Keys[0] != batchKeys(30)[10] || m.Keys[19] != batchKeys(30)[29] {
		t.Errorf("manifest should hold the newest batches, got %s … %s", m.Keys[0], m.Keys[19])
	}
}

func TestSnapshotDataset_Failures(t *testing.T) {
	tests := []struct {
		name string
		acts *TrainingActivities
		in   SnapshotDatasetInput
	}{
		{"no dataset store", &TrainingActivities{Artifacts: memArtifacts{}}, SnapshotDatasetInput{JobID: "j"}},
		{"no artifact store", &TrainingActivities{Dataset: fakeDataset{keys: batchKeys(5)}}, SnapshotDatasetInput{JobID: "j"}},
		{"not enough data", &TrainingActivities{Dataset: fakeDataset{keys: batchKeys(5)}, Artifacts: memArtifacts{}},
			SnapshotDatasetInput{JobID: "j", MinTransitions: 1000}},
		{"empty bucket", &TrainingActivities{Dataset: fakeDataset{}, Artifacts: memArtifacts{}}, SnapshotDatasetInput{JobID: "j"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := newActivityEnv(t, tt.acts)
			_, err := env.ExecuteActivity(tt.acts.SnapshotDataset, tt.in)
			requireNonRetryable(t, err)
		})
	}
}

func TestSnapshotDataset_ListErrorIsRetryable(t *testing.T) {
	acts := &TrainingActivities{Dataset: fakeDataset{err: errors.New("minio down")}, Artifacts: memArtifacts{}}
	env := newActivityEnv(t, acts)
	if _, err := env.ExecuteActivity(acts.SnapshotDataset, SnapshotDatasetInput{JobID: "j"}); err == nil {
		t.Fatal("expected error")
	}
}
