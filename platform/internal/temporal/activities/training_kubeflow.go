package activities

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"log/slog"
	"regexp"
	"strconv"
	"strings"
	"time"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"

	"github.com/dimuthu/robot-fleet/internal/kube"
	"github.com/dimuthu/robot-fleet/internal/kubeflow"
)

// Kubernetes object-name limits.
const (
	maxKubeNameLen  = 63 // DNS-1123 label
	maxKatibNameLen = 40 // Katib's experiment-name limit (trial names are derived from it)
	nameHashLen     = 8  // hex chars of the uniqueness hash on truncated names
)

// Default Katib search settings (small: this is CPU-only local hardware).
const (
	DefaultHPOAlgorithm          = "tpe"
	DefaultHPOMaxTrials          = 4
	DefaultHPOParallelTrials     = 2
	DefaultHPOTrialGradientSteps = 2_000
	DefaultHPOEvalEpisodes       = 5
)

// KubeflowRunner submits and tracks Kubeflow TrainJobs and Katib Experiments
// (implemented by *kubeflow.Runner).
type KubeflowRunner interface {
	SubmitTrainJob(ctx context.Context, spec kubeflow.TrainJobSpec) (string, error)
	TrainJobStatus(ctx context.Context, name string) (kubeflow.JobStatus, error)
	SubmitExperiment(ctx context.Context, spec kubeflow.ExperimentSpec) (string, error)
	ExperimentStatus(ctx context.Context, name string) (kubeflow.ExperimentStatus, error)
}

// HPOSettings configures Katib hyperparameter searches.
type HPOSettings struct {
	Algorithm          string
	MaxTrials          int
	ParallelTrials     int
	TrialGradientSteps int // offline updates per trial
	EvalEpisodes       int // test-track episodes per trial (the objective)
}

// DefaultHPOSettings returns the default Katib search settings.
func DefaultHPOSettings() HPOSettings {
	return HPOSettings{
		Algorithm:          DefaultHPOAlgorithm,
		MaxTrials:          DefaultHPOMaxTrials,
		ParallelTrials:     DefaultHPOParallelTrials,
		TrialGradientSteps: DefaultHPOTrialGradientSteps,
		EvalEpisodes:       DefaultHPOEvalEpisodes,
	}
}

var invalidNameChars = regexp.MustCompile(`[^a-z0-9-]+`)

// kubeName builds a deterministic DNS-1123 name (prefix-jobID, or just jobID)
// of at most maxLen chars, so a retried activity finds the object its previous
// attempt created. Long names are truncated and suffixed with a hash of the
// full name to keep them unique.
func kubeName(prefix, jobID string, maxLen int) string {
	full := jobID
	if prefix != "" {
		full = prefix + "-" + jobID
	}
	name := strings.Trim(invalidNameChars.ReplaceAllString(strings.ToLower(full), "-"), "-")
	if len(name) <= maxLen {
		return name
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(full)) // hash.Hash writes never fail
	suffix := fmt.Sprintf("%0*x", nameHashLen, h.Sum32())
	return strings.TrimRight(name[:maxLen-nameHashLen-1], "-") + "-" + suffix
}

// submitKubeflowRun trains offline with a TrainJob (train_offline.py, no
// simulator), then evaluates the uploaded policy on the MuJoCo test track with
// a second TrainJob that writes evaluations/<job-id>/eval_results.json.
func (a *TrainingActivities) submitKubeflowRun(ctx context.Context, input SubmitTrainingRunInput) (*SubmitTrainingRunOutput, error) {
	if a.Kubeflow == nil {
		return nil, temporal.NewNonRetryableApplicationError(
			"kubeflow backend not configured (set KUBEFLOW_ENABLED=true)", ErrTypeTrainingNotConfigured, nil)
	}
	if input.DatasetManifest == "" {
		return nil, temporal.NewNonRetryableApplicationError(
			"training run needs a dataset manifest (run SnapshotDataset first)", ErrTypeUnsupportedTraining, nil)
	}
	hpArgs, err := kubeflow.HyperparameterArgs(input.Hyperparameters)
	if err != nil {
		return nil, temporal.NewNonRetryableApplicationError(err.Error(), ErrTypeUnsupportedTraining, err)
	}

	labels := map[string]string{LabelJobID: kubeName("", input.JobID, maxKubeNameLen)}
	trainCmd := []string{
		"python", "train_offline.py",
		"--job-id=" + input.JobID,
		"--dataset-manifest=" + input.DatasetManifest,
		"--output-dir=/tmp/output",
	}
	if input.GradientSteps > 0 {
		trainCmd = append(trainCmd, "--gradient-steps="+strconv.Itoa(input.GradientSteps))
	}
	trainCmd = append(trainCmd, hpArgs...)

	trainName := kubeName("train", input.JobID, maxKubeNameLen)
	if err := a.runTrainJob(ctx, kubeflow.TrainJobSpec{Name: trainName, Command: trainCmd, Labels: labels}); err != nil {
		return nil, err
	}

	evalCmd := []string{
		"python", "evaluate_policy.py",
		"--eval-id=" + input.JobID,
		fmt.Sprintf("--model-path=s3://%s/%s", a.ModelsBucket, ModelArtifactKey(input.JobID)),
		"--env-id=" + EvalEnvID,
		"--output-dir=/tmp/eval",
	}
	if err := a.runTrainJob(ctx, kubeflow.TrainJobSpec{Name: kubeName("eval", input.JobID, maxKubeNameLen), Command: evalCmd, Labels: labels}); err != nil {
		return nil, err
	}

	return &SubmitTrainingRunOutput{RunID: trainName, Status: "Succeeded", ModelPath: ModelArtifactKey(input.JobID)}, nil
}

// runTrainJob creates (or, on retry, reuses) a TrainJob and waits for it.
func (a *TrainingActivities) runTrainJob(ctx context.Context, spec kubeflow.TrainJobSpec) error {
	if _, err := a.Kubeflow.SubmitTrainJob(ctx, spec); err != nil && !errors.Is(err, kube.ErrAlreadyExists) {
		return fmt.Errorf("submit trainjob %s: %w", spec.Name, err)
	}
	slog.Info("trainjob running", "name", spec.Name)

	_, err := pollUntilDone(ctx, a.PollInterval, spec.Name, func() (bool, error) {
		st, err := a.Kubeflow.TrainJobStatus(ctx, spec.Name)
		if err != nil {
			return false, err
		}
		if st.Done && !st.Succeeded {
			return true, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("trainjob %s failed: %s", spec.Name, st.Message), ErrTypeTrainingRunFailed, nil)
		}
		return st.Done, nil
	})
	return err
}

// HyperparameterSearchInput is the input for RunHyperparameterSearch.
type HyperparameterSearchInput struct {
	JobID           string `json:"job_id"`
	DatasetManifest string `json:"dataset_manifest"`
}

// HyperparameterSearchOutput is Katib's best trial.
type HyperparameterSearchOutput struct {
	Experiment    string            `json:"experiment"`
	BestTrial     string            `json:"best_trial"`
	BestParams    map[string]string `json:"best_params"`
	BestObjective float64           `json:"best_objective"`
}

// RunHyperparameterSearch runs a Katib Experiment of short offline trials on
// the dataset snapshot, each scored on the test track, and returns the best
// trial's hyperparameters.
func (a *TrainingActivities) RunHyperparameterSearch(ctx context.Context, input HyperparameterSearchInput) (*HyperparameterSearchOutput, error) {
	if a.Kubeflow == nil {
		return nil, temporal.NewNonRetryableApplicationError(
			"katib not configured (set KUBEFLOW_ENABLED=true)", ErrTypeTrainingNotConfigured, nil)
	}
	if input.DatasetManifest == "" {
		return nil, temporal.NewNonRetryableApplicationError(
			"hyperparameter search needs a dataset manifest", ErrTypeUnsupportedTraining, nil)
	}

	name := kubeName("hpo", input.JobID, maxKatibNameLen)
	spec := kubeflow.ExperimentSpec{
		Name:               name,
		Algorithm:          a.HPO.Algorithm,
		MaxTrials:          a.HPO.MaxTrials,
		ParallelTrials:     a.HPO.ParallelTrials,
		DatasetManifest:    input.DatasetManifest,
		TrialGradientSteps: a.HPO.TrialGradientSteps,
		EvalEpisodes:       a.HPO.EvalEpisodes,
		Labels:             map[string]string{LabelJobID: kubeName("", input.JobID, maxKubeNameLen)},
	}
	if _, err := a.Kubeflow.SubmitExperiment(ctx, spec); err != nil && !errors.Is(err, kube.ErrAlreadyExists) {
		return nil, fmt.Errorf("submit katib experiment %s: %w", name, err)
	}
	slog.Info("katib experiment running", "name", name, "algorithm", spec.Algorithm, "max_trials", spec.MaxTrials)

	var final kubeflow.ExperimentStatus
	if _, err := pollUntilDone(ctx, a.PollInterval, name, func() (bool, error) {
		st, err := a.Kubeflow.ExperimentStatus(ctx, name)
		if err != nil {
			return false, err
		}
		final = st
		return st.Done, nil
	}); err != nil {
		return nil, err
	}

	if !final.Succeeded {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("katib experiment %s failed: %s", name, final.Message), ErrTypeTrainingRunFailed, nil)
	}
	if len(final.BestParams) == 0 {
		return nil, temporal.NewNonRetryableApplicationError(
			fmt.Sprintf("katib experiment %s finished without a best trial", name), ErrTypeTrainingRunFailed, nil)
	}
	slog.Info("katib search complete", "name", name, "best_trial", final.BestTrial,
		"best_objective", final.BestObjective, "best_params", final.BestParams)
	return &HyperparameterSearchOutput{
		Experiment:    name,
		BestTrial:     final.BestTrial,
		BestParams:    final.BestParams,
		BestObjective: final.BestObjective,
	}, nil
}

// pollUntilDone calls check every interval (heartbeating name) until it
// reports done or returns an error. Transient errors are tolerated up to
// MaxConsecutivePollErrors; Temporal application errors stop polling at once.
func pollUntilDone(ctx context.Context, interval time.Duration, name string, check func() (bool, error)) (bool, error) {
	if interval <= 0 {
		interval = DefaultPollInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	consecutiveErrs := 0
	for {
		done, err := check()
		var appErr *temporal.ApplicationError
		switch {
		case errors.As(err, &appErr):
			return false, err
		case errors.Is(err, kube.ErrNotFound):
			return false, temporal.NewNonRetryableApplicationError(
				fmt.Sprintf("%s disappeared", name), ErrTypeTrainingRunFailed, err)
		case err != nil:
			consecutiveErrs++
			slog.Warn("poll failed", "name", name, "attempt", consecutiveErrs, "error", err)
			if consecutiveErrs >= MaxConsecutivePollErrors {
				return false, fmt.Errorf("poll %s: %w", name, err)
			}
		case done:
			return true, nil
		default:
			consecutiveErrs = 0
		}
		activity.RecordHeartbeat(ctx, name)

		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-ticker.C:
		}
	}
}
