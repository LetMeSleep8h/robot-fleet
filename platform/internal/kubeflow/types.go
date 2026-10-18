// Package kubeflow submits FleetOS training to Kubeflow: Trainer v2 TrainJobs
// for training/evaluation runs and Katib Experiments for hyperparameter search.
package kubeflow

import (
	"context"

	"github.com/dimuthu/robot-fleet/internal/kube"
)

// Kubeflow API resources used by FleetOS.
var (
	TrainJobs   = kube.Resource{Group: "trainer.kubeflow.org", Version: "v1alpha1", Resource: "trainjobs"}
	Experiments = kube.Resource{Group: "kubeflow.org", Version: "v1beta1", Resource: "experiments"}
)

// ObjectiveMetric is printed by train_offline.py --eval-episodes
// ("eval_mean_reward=<x>", measured on the MuJoCo test track) and maximized by Katib.
const ObjectiveMetric = "eval_mean_reward"

// API is the subset of the Kubernetes client the runner needs.
type API interface {
	Create(ctx context.Context, r kube.Resource, namespace string, obj map[string]any) (map[string]any, error)
	Get(ctx context.Context, r kube.Resource, namespace, name string) (map[string]any, error)
}

// Config locates FleetOS training resources in the cluster.
type Config struct {
	Namespace      string // namespace for TrainJobs and Experiments, e.g. "fleetos-training"
	Runtime        string // ClusterTrainingRuntime name, e.g. "fleetos-sb3"
	Image          string // training image used by Katib trials
	TrainingSecret string // secret with S3 credentials, mounted into trials (envFrom)
}

// TrainJobSpec describes one container run on the FleetOS training runtime.
type TrainJobSpec struct {
	Name    string
	Command []string // full argv, e.g. python train_locomotion.py --job-id=...
	Labels  map[string]string
}

// JobStatus is the terminal state of a TrainJob (zero value = still running).
type JobStatus struct {
	Done      bool
	Succeeded bool
	Message   string
}

// ExperimentSpec describes a Katib hyperparameter search over SearchSpace.
// Each trial trains offline on the dataset snapshot, then scores the policy on
// a few MuJoCo test-track episodes.
type ExperimentSpec struct {
	Name               string
	Algorithm          string // Katib algorithm, e.g. "tpe", "random", "bayesianoptimization"
	MaxTrials          int
	ParallelTrials     int
	DatasetManifest    string // S3 key of the dataset snapshot (models bucket)
	TrialGradientSteps int    // offline updates per trial (short; only the metric matters)
	EvalEpisodes       int    // test-track episodes per trial for the objective
	Labels             map[string]string
}

// ExperimentStatus is the state of a Katib Experiment.
type ExperimentStatus struct {
	Done          bool
	Succeeded     bool
	Message       string
	BestTrial     string
	BestParams    map[string]string // hyperparameter name → value
	BestObjective float64
}

// Hyperparameter is one dimension of the search space.
type Hyperparameter struct {
	Name       string   // Katib parameter name and results key, e.g. "learning_rate"
	TrialParam string   // trial template placeholder, e.g. "learningRate"
	Flag       string   // train_offline.py flag, e.g. "--learning-rate"
	Type       string   // Katib parameterType: double | int | categorical
	Min, Max   string   // for double/int
	List       []string // for categorical
}

// SearchSpace is the offline CQL hyperparameter space train_offline.py exposes.
var SearchSpace = []Hyperparameter{
	{Name: "learning_rate", TrialParam: "learningRate", Flag: "--learning-rate", Type: "double", Min: "0.0001", Max: "0.001"},
	{Name: "cql_alpha", TrialParam: "cqlAlpha", Flag: "--cql-alpha", Type: "double", Min: "0.1", Max: "5.0"},
	{Name: "batch_size", TrialParam: "batchSize", Flag: "--batch-size", Type: "categorical", List: []string{"128", "256", "512"}},
}
