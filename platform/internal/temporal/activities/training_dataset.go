package activities

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"go.temporal.io/sdk/temporal"

	"github.com/dimuthu/robot-fleet/internal/experience"
)

// ErrTypeInsufficientData means the fleet hasn't collected enough experience.
const ErrTypeInsufficientData = "InsufficientData"

// DatasetManifestKey is where a job's dataset snapshot is written (models bucket).
func DatasetManifestKey(jobID string) string {
	return fmt.Sprintf("datasets/%s/manifest.json", jobID)
}

// DatasetManifest pins the exact experience batches a training run uses, so
// every model can be traced back to its data. Read by train_offline.py.
type DatasetManifest struct {
	JobID                string   `json:"job_id"`
	Bucket               string   `json:"bucket"`
	Prefix               string   `json:"prefix"`
	Keys                 []string `json:"keys"`
	Files                int      `json:"files"`
	EstimatedTransitions int      `json:"estimated_transitions"`
	CreatedAt            string   `json:"created_at"`
}

// SnapshotDatasetInput is the input for SnapshotDataset.
type SnapshotDatasetInput struct {
	JobID          string `json:"job_id"`
	MaxFiles       int    `json:"max_files"`       // newest N batches; 0 = all
	MinTransitions int    `json:"min_transitions"` // refuse to train on less
}

// SnapshotDatasetOutput describes the snapshot.
type SnapshotDatasetOutput struct {
	ManifestKey          string `json:"manifest_key"`
	Files                int    `json:"files"`
	EstimatedTransitions int    `json:"estimated_transitions"`
}

// SnapshotDataset lists the fleet's experience batches, keeps the newest
// MaxFiles, and writes a manifest pinning them for this training job.
func (a *TrainingActivities) SnapshotDataset(ctx context.Context, input SnapshotDatasetInput) (*SnapshotDatasetOutput, error) {
	if a.Dataset == nil || a.Artifacts == nil {
		return nil, temporal.NewNonRetryableApplicationError(
			"experience dataset store not configured", ErrTypeTrainingNotConfigured, nil)
	}

	keys, err := a.Dataset.List(ctx, experience.Prefix, 0)
	if err != nil {
		return nil, fmt.Errorf("list experience batches: %w", err)
	}
	// Keys are time-partitioned (training_data/YYYY/MM/DD/HH/...), so
	// lexicographic order is chronological; keep the newest batches.
	if input.MaxFiles > 0 && len(keys) > input.MaxFiles {
		keys = keys[len(keys)-input.MaxFiles:]
	}
	estimated := len(keys) * experience.TransitionsPerBatch
	if len(keys) == 0 || estimated < input.MinTransitions {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("not enough fleet experience: %d batches (~%d transitions), need %d transitions",
				len(keys), estimated, input.MinTransitions),
			ErrTypeInsufficientData, nil)
	}

	manifest := DatasetManifest{
		JobID:                input.JobID,
		Bucket:               a.DatasetBucket,
		Prefix:               experience.Prefix,
		Keys:                 keys,
		Files:                len(keys),
		EstimatedTransitions: estimated,
		CreatedAt:            time.Now().UTC().Format(time.RFC3339),
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return nil, fmt.Errorf("encode dataset manifest: %w", err)
	}
	key := DatasetManifestKey(input.JobID)
	if err := a.Artifacts.Put(ctx, key, data, "application/json"); err != nil {
		return nil, fmt.Errorf("write dataset manifest %s: %w", key, err)
	}

	slog.Info("dataset snapshot written", "job_id", input.JobID, "manifest", key,
		"files", len(keys), "estimated_transitions", estimated)
	return &SnapshotDatasetOutput{ManifestKey: key, Files: len(keys), EstimatedTransitions: estimated}, nil
}
