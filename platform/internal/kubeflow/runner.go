package kubeflow

import (
	"context"
	"fmt"
	"sort"
	"strconv"

	"github.com/dimuthu/robot-fleet/internal/kube"
)

// Trial resources: short CPU runs that only report the objective metric.
const (
	trialCPURequest    = "1"
	trialCPULimit      = "2"
	trialMemoryRequest = "2Gi"
	trialMemoryLimit   = "3Gi"
	trialContainer     = "training"
)

// Runner submits and tracks Kubeflow TrainJobs and Katib Experiments.
type Runner struct {
	api API
	cfg Config
}

// NewRunner creates a Runner.
func NewRunner(api API, cfg Config) *Runner {
	return &Runner{api: api, cfg: cfg}
}

// SubmitTrainJob creates a TrainJob on the FleetOS training runtime.
func (r *Runner) SubmitTrainJob(ctx context.Context, spec TrainJobSpec) (string, error) {
	obj := map[string]any{
		"apiVersion": "trainer.kubeflow.org/v1alpha1",
		"kind":       "TrainJob",
		"metadata":   map[string]any{"name": spec.Name, "namespace": r.cfg.Namespace, "labels": spec.Labels},
		"spec": map[string]any{
			"runtimeRef": map[string]any{
				"name":     r.cfg.Runtime,
				"kind":     "ClusterTrainingRuntime",
				"apiGroup": "trainer.kubeflow.org",
			},
			"trainer": map[string]any{"command": spec.Command},
		},
	}
	if _, err := r.api.Create(ctx, TrainJobs, r.cfg.Namespace, obj); err != nil {
		return "", fmt.Errorf("submit trainjob %s: %w", spec.Name, err)
	}
	return spec.Name, nil
}

// TrainJobStatus reports whether a TrainJob has completed or failed.
func (r *Runner) TrainJobStatus(ctx context.Context, name string) (JobStatus, error) {
	obj, err := r.api.Get(ctx, TrainJobs, r.cfg.Namespace, name)
	if err != nil {
		return JobStatus{}, fmt.Errorf("trainjob status %s: %w", name, err)
	}
	if c, ok := kube.Condition(obj, "Failed"); ok && c.Status == "True" {
		return JobStatus{Done: true, Message: c.Message}, nil
	}
	if c, ok := kube.Condition(obj, "Complete"); ok && c.Status == "True" {
		return JobStatus{Done: true, Succeeded: true, Message: c.Message}, nil
	}
	return JobStatus{}, nil
}

// SubmitExperiment creates a Katib Experiment that searches SearchSpace by
// running short train_locomotion.py trials and maximizing ObjectiveMetric.
func (r *Runner) SubmitExperiment(ctx context.Context, spec ExperimentSpec) (string, error) {
	params := make([]map[string]any, 0, len(SearchSpace))
	trialParams := make([]map[string]any, 0, len(SearchSpace)+1)
	command := []string{
		"python", "train_locomotion.py",
		"--job-id=${trialParameters.trialName}",
		"--timesteps=" + strconv.Itoa(spec.TrialTimesteps),
		"--output-dir=/tmp/output",
	}
	if spec.EnvID != "" {
		command = append(command, "--env-id="+spec.EnvID)
	}
	for _, hp := range SearchSpace {
		space := map[string]any{}
		if hp.Type == "categorical" {
			space["list"] = hp.List
		} else {
			space["min"], space["max"] = hp.Min, hp.Max
		}
		params = append(params, map[string]any{"name": hp.Name, "parameterType": hp.Type, "feasibleSpace": space})
		trialParams = append(trialParams, map[string]any{"name": hp.TrialParam, "reference": hp.Name})
		command = append(command, fmt.Sprintf("%s=${trialParameters.%s}", hp.Flag, hp.TrialParam))
	}
	trialParams = append(trialParams, map[string]any{"name": "trialName", "reference": "${trialSpec.Name}"})
	// Trials only report the metric: no artifact uploads, no callbacks. Set via
	// flags because Katib's template validation drops empty env values.
	command = append(command, "--s3-endpoint=", "--callback-url=")

	obj := map[string]any{
		"apiVersion": "kubeflow.org/v1beta1",
		"kind":       "Experiment",
		"metadata":   map[string]any{"name": spec.Name, "namespace": r.cfg.Namespace, "labels": spec.Labels},
		"spec": map[string]any{
			"objective":            map[string]any{"type": "maximize", "objectiveMetricName": ObjectiveMetric},
			"algorithm":            map[string]any{"algorithmName": spec.Algorithm},
			"maxTrialCount":        spec.MaxTrials,
			"parallelTrialCount":   spec.ParallelTrials,
			"maxFailedTrialCount":  spec.MaxTrials,
			"parameters":           params,
			"metricsCollectorSpec": map[string]any{"collector": map[string]any{"kind": "StdOut"}},
			"trialTemplate": map[string]any{
				"primaryContainerName": trialContainer,
				"trialParameters":      trialParams,
				"trialSpec": map[string]any{
					"apiVersion": "batch/v1",
					"kind":       "Job",
					"spec": map[string]any{
						"backoffLimit": 0,
						"template": map[string]any{
							"spec": map[string]any{
								"restartPolicy": "Never",
								"containers": []map[string]any{{
									"name":            trialContainer,
									"image":           r.cfg.Image,
									"imagePullPolicy": "IfNotPresent",
									"command":         command,
									"resources": map[string]any{
										"requests": map[string]any{"cpu": trialCPURequest, "memory": trialMemoryRequest},
										"limits":   map[string]any{"cpu": trialCPULimit, "memory": trialMemoryLimit},
									},
								}},
							},
						},
					},
				},
			},
		},
	}
	if _, err := r.api.Create(ctx, Experiments, r.cfg.Namespace, obj); err != nil {
		return "", fmt.Errorf("submit experiment %s: %w", spec.Name, err)
	}
	return spec.Name, nil
}

// ExperimentStatus reports an Experiment's state and its best trial so far.
func (r *Runner) ExperimentStatus(ctx context.Context, name string) (ExperimentStatus, error) {
	obj, err := r.api.Get(ctx, Experiments, r.cfg.Namespace, name)
	if err != nil {
		return ExperimentStatus{}, fmt.Errorf("experiment status %s: %w", name, err)
	}
	var st ExperimentStatus
	if c, ok := kube.Condition(obj, "Failed"); ok && c.Status == "True" {
		st.Done, st.Message = true, c.Message
	} else if c, ok := kube.Condition(obj, "Succeeded"); ok && c.Status == "True" {
		st.Done, st.Succeeded, st.Message = true, true, c.Message
	}

	status, _ := obj["status"].(map[string]any)
	best, _ := status["currentOptimalTrial"].(map[string]any)
	st.BestTrial, _ = best["bestTrialName"].(string)
	assignments, _ := best["parameterAssignments"].([]any)
	if len(assignments) > 0 {
		st.BestParams = make(map[string]string, len(assignments))
		for _, raw := range assignments {
			a, _ := raw.(map[string]any)
			n, _ := a["name"].(string)
			v, _ := a["value"].(string)
			st.BestParams[n] = v
		}
	}
	observation, _ := best["observation"].(map[string]any)
	metrics, _ := observation["metrics"].([]any)
	for _, raw := range metrics {
		m, _ := raw.(map[string]any)
		if n, _ := m["name"].(string); n == ObjectiveMetric {
			latest, _ := m["latest"].(string)
			if f, err := strconv.ParseFloat(latest, 64); err == nil {
				st.BestObjective = f
			}
		}
	}
	return st, nil
}

// HyperparameterArgs converts Katib's best parameter assignments into
// train_locomotion.py flags, sorted for deterministic commands.
func HyperparameterArgs(params map[string]string) ([]string, error) {
	flags := make(map[string]string, len(SearchSpace))
	for _, hp := range SearchSpace {
		flags[hp.Name] = hp.Flag
	}
	args := make([]string, 0, len(params))
	for name, value := range params {
		flag, ok := flags[name]
		if !ok {
			return nil, fmt.Errorf("unknown hyperparameter %q", name)
		}
		args = append(args, flag+"="+value)
	}
	sort.Strings(args)
	return args, nil
}
