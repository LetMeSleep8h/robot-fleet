package config

import (
	"testing"
)

func TestLoad_Defaults(t *testing.T) {
	cfg := Load()

	if cfg.ServiceName != "fleetos" {
		t.Errorf("expected fleetos, got %s", cfg.ServiceName)
	}
	if cfg.Environment != "dev" {
		t.Errorf("expected dev, got %s", cfg.Environment)
	}
	if cfg.GRPCPort != 50051 {
		t.Errorf("expected 50051, got %d", cfg.GRPCPort)
	}
	if cfg.HTTPPort != 8080 {
		t.Errorf("expected 8080, got %d", cfg.HTTPPort)
	}
	if len(cfg.KafkaBrokers) != 1 || cfg.KafkaBrokers[0] != "localhost:9092" {
		t.Errorf("unexpected kafka brokers: %v", cfg.KafkaBrokers)
	}
	if cfg.KafkaTelemetryTopic != "robot.telemetry" {
		t.Errorf("expected robot.telemetry, got %s", cfg.KafkaTelemetryTopic)
	}
	if cfg.RedisAddr != "localhost:6379" {
		t.Errorf("expected localhost:6379, got %s", cfg.RedisAddr)
	}
	if cfg.RateLimitRPS != 100 {
		t.Errorf("expected 100, got %d", cfg.RateLimitRPS)
	}
}

func TestLoad_EnvOverrides(t *testing.T) {
	t.Setenv("SERVICE_NAME", "custom-svc")
	t.Setenv("HTTP_PORT", "9999")
	t.Setenv("RATE_LIMIT_RPS", "500")

	cfg := Load()

	if cfg.ServiceName != "custom-svc" {
		t.Errorf("expected custom-svc, got %s", cfg.ServiceName)
	}
	if cfg.HTTPPort != 9999 {
		t.Errorf("expected 9999, got %d", cfg.HTTPPort)
	}
	if cfg.RateLimitRPS != 500 {
		t.Errorf("expected 500, got %d", cfg.RateLimitRPS)
	}
}

func TestLoad_InvalidIntFallback(t *testing.T) {
	t.Setenv("HTTP_PORT", "not-a-number")

	cfg := Load()

	if cfg.HTTPPort != 8080 {
		t.Errorf("expected fallback 8080, got %d", cfg.HTTPPort)
	}
}

func TestGetEnv(t *testing.T) {
	t.Setenv("TEST_KEY", "test_value")

	if v := getEnv("TEST_KEY", "default"); v != "test_value" {
		t.Errorf("expected test_value, got %s", v)
	}
	if v := getEnv("NONEXISTENT_KEY", "default"); v != "default" {
		t.Errorf("expected default, got %s", v)
	}
}

func TestGetEnvInt(t *testing.T) {
	t.Setenv("INT_KEY", "42")

	if v := getEnvInt("INT_KEY", 0); v != 42 {
		t.Errorf("expected 42, got %d", v)
	}
	if v := getEnvInt("NONEXISTENT_INT", 99); v != 99 {
		t.Errorf("expected 99, got %d", v)
	}
}

func TestLoad_MLOpsDefaults(t *testing.T) {
	cfg := Load()
	if cfg.ArgoServerURL != "" {
		t.Errorf("training must be off unless ARGO_SERVER_URL is set, got %q", cfg.ArgoServerURL)
	}
	if cfg.ArgoNamespace != "argo" || cfg.ModelsBucket != "fleetos-models" {
		t.Errorf("unexpected defaults: ns=%q bucket=%q", cfg.ArgoNamespace, cfg.ModelsBucket)
	}
	if cfg.MinEvalPassRate != 0.5 || cfg.ArgoInsecureSkipVerify {
		t.Errorf("unexpected defaults: pass=%v insecure=%v", cfg.MinEvalPassRate, cfg.ArgoInsecureSkipVerify)
	}
}

func TestLoad_MLOpsOverrides(t *testing.T) {
	t.Setenv("ARGO_SERVER_URL", "https://argo-server.argo:2746")
	t.Setenv("ARGO_NAMESPACE", "ml")
	t.Setenv("ARGO_TOKEN", "tok")
	t.Setenv("ARGO_INSECURE_SKIP_VERIFY", "true")
	t.Setenv("MODELS_BUCKET", "models")
	t.Setenv("MODEL_MIN_EVAL_PASS_RATE", "0.8")

	cfg := Load()
	if cfg.ArgoServerURL != "https://argo-server.argo:2746" || cfg.ArgoNamespace != "ml" || cfg.ArgoToken != "tok" ||
		!cfg.ArgoInsecureSkipVerify || cfg.ModelsBucket != "models" || cfg.MinEvalPassRate != 0.8 {
		t.Errorf("overrides not applied: %+v", cfg)
	}
}

func TestLoad_InvalidPassRateFallsBack(t *testing.T) {
	t.Setenv("MODEL_MIN_EVAL_PASS_RATE", "most")
	if got := Load().MinEvalPassRate; got != 0.5 {
		t.Errorf("MinEvalPassRate = %v, want default 0.5", got)
	}
}

func TestLoad_KubeflowDefaults(t *testing.T) {
	cfg := Load()
	if cfg.KubeflowEnabled {
		t.Error("kubeflow must be opt-in")
	}
	if cfg.KubeflowNamespace != "fleetos-training" || cfg.KubeflowRuntime != "fleetos-sb3" || cfg.TrainingImage != "fleetos/training:latest" {
		t.Errorf("unexpected defaults: %+v", cfg)
	}
	if cfg.HPOAlgorithm != "tpe" || cfg.HPOMaxTrials != 4 || cfg.HPOParallelTrials != 2 || cfg.HPOTrialTimesteps != 20000 {
		t.Errorf("unexpected HPO defaults: %+v", cfg)
	}
}

func TestLoad_KubeflowOverrides(t *testing.T) {
	t.Setenv("KUBEFLOW_ENABLED", "true")
	t.Setenv("KUBEFLOW_NAMESPACE", "ml")
	t.Setenv("KUBEFLOW_RUNTIME", "gpu-runtime")
	t.Setenv("TRAINING_IMAGE", "registry/training:v2")
	t.Setenv("HPO_ALGORITHM", "bayesianoptimization")
	t.Setenv("HPO_MAX_TRIALS", "20")
	t.Setenv("HPO_PARALLEL_TRIALS", "3")
	t.Setenv("HPO_TRIAL_TIMESTEPS", "100000")

	cfg := Load()
	if !cfg.KubeflowEnabled || cfg.KubeflowNamespace != "ml" || cfg.KubeflowRuntime != "gpu-runtime" || cfg.TrainingImage != "registry/training:v2" {
		t.Errorf("kubeflow overrides not applied: %+v", cfg)
	}
	if cfg.HPOAlgorithm != "bayesianoptimization" || cfg.HPOMaxTrials != 20 || cfg.HPOParallelTrials != 3 || cfg.HPOTrialTimesteps != 100000 {
		t.Errorf("HPO overrides not applied: %+v", cfg)
	}
}
