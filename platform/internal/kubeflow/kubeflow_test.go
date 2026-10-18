package kubeflow

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/kube"
)

var testCfg = Config{Namespace: "fleetos-training", Runtime: "fleetos-sb3", Image: "fleetos/training:latest", TrainingSecret: "fleetos-training"}

// fakeAPI records created objects and serves scripted Get responses.
type fakeAPI struct {
	created []created
	objects map[string]map[string]any // name → object
	err     error
}

type created struct {
	resource  kube.Resource
	namespace string
	obj       map[string]any
}

func (f *fakeAPI) Create(_ context.Context, r kube.Resource, ns string, obj map[string]any) (map[string]any, error) {
	if f.err != nil {
		return nil, f.err
	}
	f.created = append(f.created, created{r, ns, obj})
	return obj, nil
}

func (f *fakeAPI) Get(_ context.Context, _ kube.Resource, _ string, name string) (map[string]any, error) {
	if o, ok := f.objects[name]; ok {
		return o, nil
	}
	return nil, kube.ErrNotFound
}

func dig(t *testing.T, obj map[string]any, path ...string) any {
	t.Helper()
	var cur any = obj
	for _, p := range path {
		m, ok := cur.(map[string]any)
		if !ok {
			t.Fatalf("path %v: %q is not an object", path, p)
		}
		cur = m[p]
	}
	return cur
}

func TestSubmitTrainJob_BuildsTrainJobOnFleetOSRuntime(t *testing.T) {
	api := &fakeAPI{}
	r := NewRunner(api, testCfg)

	name, err := r.SubmitTrainJob(context.Background(), TrainJobSpec{
		Name:    "train-job-1",
		Command: []string{"python", "train_locomotion.py", "--job-id=job-1"},
		Labels:  map[string]string{"fleetos.io/job-id": "job-1"},
	})
	if err != nil {
		t.Fatalf("SubmitTrainJob: %v", err)
	}
	if name != "train-job-1" {
		t.Errorf("name = %q", name)
	}
	c := api.created[0]
	if c.resource != TrainJobs || c.namespace != "fleetos-training" {
		t.Errorf("created %v in %q", c.resource, c.namespace)
	}
	obj := c.obj
	if obj["kind"] != "TrainJob" || obj["apiVersion"] != "trainer.kubeflow.org/v1alpha1" {
		t.Errorf("kind/apiVersion = %v/%v", obj["kind"], obj["apiVersion"])
	}
	if dig(t, obj, "spec", "runtimeRef", "name") != "fleetos-sb3" {
		t.Errorf("runtimeRef = %v", dig(t, obj, "spec", "runtimeRef"))
	}
	if got := dig(t, obj, "spec", "trainer", "command"); !reflect.DeepEqual(got, []string{"python", "train_locomotion.py", "--job-id=job-1"}) {
		t.Errorf("command = %v", got)
	}
	if dig(t, obj, "metadata", "labels").(map[string]string)["fleetos.io/job-id"] != "job-1" {
		t.Error("labels not set")
	}
}

func TestTrainJobStatus(t *testing.T) {
	cond := func(typ, msg string) map[string]any {
		return map[string]any{"status": map[string]any{"conditions": []any{
			map[string]any{"type": typ, "status": "True", "message": msg},
		}}}
	}
	api := &fakeAPI{objects: map[string]map[string]any{
		"done":    cond("Complete", "jobs completed"),
		"failed":  cond("Failed", "OOMKilled"),
		"running": {"status": map[string]any{}},
	}}
	r := NewRunner(api, testCfg)

	for name, want := range map[string]JobStatus{
		"done":    {Done: true, Succeeded: true, Message: "jobs completed"},
		"failed":  {Done: true, Succeeded: false, Message: "OOMKilled"},
		"running": {},
	} {
		got, err := r.TrainJobStatus(context.Background(), name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got != want {
			t.Errorf("%s: status = %+v, want %+v", name, got, want)
		}
	}
	if _, err := r.TrainJobStatus(context.Background(), "gone"); !errors.Is(err, kube.ErrNotFound) {
		t.Errorf("missing TrainJob: err = %v", err)
	}
}

func TestSubmitExperiment_BuildsOfflineKatibSearch(t *testing.T) {
	api := &fakeAPI{}
	r := NewRunner(api, testCfg)

	if _, err := r.SubmitExperiment(context.Background(), ExperimentSpec{
		Name: "hpo-job-1", Algorithm: "tpe", MaxTrials: 4, ParallelTrials: 2,
		DatasetManifest: "datasets/job-1/manifest.json", TrialGradientSteps: 2000, EvalEpisodes: 5,
	}); err != nil {
		t.Fatalf("SubmitExperiment: %v", err)
	}
	c := api.created[0]
	if c.resource != Experiments {
		t.Errorf("resource = %v", c.resource)
	}
	obj := c.obj
	if dig(t, obj, "spec", "objective", "objectiveMetricName") != ObjectiveMetric || dig(t, obj, "spec", "objective", "type") != "maximize" {
		t.Errorf("objective = %v", dig(t, obj, "spec", "objective"))
	}
	if dig(t, obj, "spec", "algorithm", "algorithmName") != "tpe" || dig(t, obj, "spec", "maxTrialCount") != 4 || dig(t, obj, "spec", "parallelTrialCount") != 2 {
		t.Errorf("search settings wrong: %v", obj["spec"])
	}
	params := dig(t, obj, "spec", "parameters").([]map[string]any)
	if len(params) != len(SearchSpace) {
		t.Errorf("expected %d parameters, got %d", len(SearchSpace), len(params))
	}

	containers := dig(t, obj, "spec", "trialTemplate", "trialSpec", "spec", "template", "spec", "containers").([]map[string]any)
	cmd := strings.Join(containers[0]["command"].([]string), " ")
	for _, want := range []string{
		"train_offline.py", "--dataset-manifest=datasets/job-1/manifest.json", "--gradient-steps=2000", "--eval-episodes=5",
		"--learning-rate=${trialParameters.learningRate}", "--cql-alpha=${trialParameters.cqlAlpha}",
		// Trials only report the objective: no uploads, no callbacks.
		"--no-upload",
	} {
		if !strings.Contains(cmd, want) {
			t.Errorf("trial command %q missing %q", cmd, want)
		}
	}
	if strings.Contains(cmd, "train_locomotion.py") {
		t.Error("trials must train offline, not in the simulator")
	}
	// Trials read the dataset from S3, so they need the training secret.
	envFrom := containers[0]["envFrom"].([]map[string]any)
	if envFrom[0]["secretRef"].(map[string]any)["name"] != "fleetos-training" {
		t.Errorf("envFrom = %v", envFrom)
	}
	if _, hasEnv := containers[0]["env"]; hasEnv {
		t.Error("trial containers must not set env (Katib drops empty env values)")
	}
}

func TestExperimentStatus_ReadsBestTrial(t *testing.T) {
	api := &fakeAPI{objects: map[string]map[string]any{
		"hpo-1": {"status": map[string]any{
			"conditions": []any{map[string]any{"type": "Succeeded", "status": "True", "message": "max trials reached"}},
			"currentOptimalTrial": map[string]any{
				"bestTrialName": "hpo-1-abc",
				"parameterAssignments": []any{
					map[string]any{"name": "learning_rate", "value": "0.0012"},
					map[string]any{"name": "n_steps", "value": "1024"},
				},
				"observation": map[string]any{"metrics": []any{
					map[string]any{"name": "eval_mean_reward", "latest": "17.25", "max": "17.25"},
				}},
			},
		}},
		"hpo-running": {"status": map[string]any{}},
	}}
	r := NewRunner(api, testCfg)

	got, err := r.ExperimentStatus(context.Background(), "hpo-1")
	if err != nil {
		t.Fatalf("ExperimentStatus: %v", err)
	}
	if !got.Done || !got.Succeeded || got.BestTrial != "hpo-1-abc" || got.BestObjective != 17.25 {
		t.Errorf("unexpected status %+v", got)
	}
	if !reflect.DeepEqual(got.BestParams, map[string]string{"learning_rate": "0.0012", "n_steps": "1024"}) {
		t.Errorf("best params = %v", got.BestParams)
	}

	running, _ := r.ExperimentStatus(context.Background(), "hpo-running")
	if running.Done {
		t.Error("experiment without terminal condition must not be done")
	}
}

func TestHyperparameterArgs(t *testing.T) {
	args, err := HyperparameterArgs(map[string]string{"cql_alpha": "2.5", "learning_rate": "0.0012"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(args, []string{"--cql-alpha=2.5", "--learning-rate=0.0012"}) {
		t.Errorf("args = %v", args)
	}
	if _, err := HyperparameterArgs(map[string]string{"momentum": "0.9"}); err == nil {
		t.Error("unknown hyperparameter must be rejected")
	}
}
