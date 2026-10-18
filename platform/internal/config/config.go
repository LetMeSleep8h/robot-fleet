package config

import (
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/dimuthu/robot-fleet/internal/modelgate"
)

// Config holds all configuration for FleetOS services.
type Config struct {
	// Service identity
	ServiceName string
	Environment string // dev, staging, prod

	// gRPC server
	GRPCPort int

	// HTTP/REST server
	HTTPPort int

	// Kafka
	KafkaBrokers              []string
	KafkaTelemetryTopic       string
	KafkaCommandTopic         string
	KafkaCommandDispatchTopic string // Temporal → ingestion (commands to robots)
	KafkaCommandAcksTopic     string // ingestion → Temporal (robot acknowledgments)
	KafkaDeploymentEventTopic string // Temporal activities → SSE consumers
	KafkaGroupID              string

	// PostgreSQL
	PostgresDSN string

	// Redis
	RedisAddr     string
	RedisPassword string
	RedisDB       int

	// Inference
	InferenceEndpoint string
	InferenceTimeout  time.Duration

	// S3 / MinIO
	S3Endpoint  string
	S3Bucket    string
	S3AccessKey string
	S3SecretKey string
	S3UseSSL    bool

	// Auth
	JWTSecret    string
	OAuth2Issuer string

	// Rate limiting
	RateLimitRPS   int
	RateLimitBurst int

	// Temporal (control plane)
	TemporalHostPort  string
	TemporalNamespace string
	TemporalEnabled   bool

	// Training (Argo Workflows; empty ArgoServerURL = training disabled)
	ArgoServerURL          string
	ArgoNamespace          string
	ArgoToken              string
	ArgoInsecureSkipVerify bool
	ModelsBucket           string
	TelemetryBucket        string  // fleet experience (training_data/) written by the processor
	MinEvalPassRate        float64 // promotion gate: minimum eval scenario pass rate

	// Kubeflow (Trainer TrainJobs + Katib HPO); opt-in
	KubeflowEnabled        bool
	KubeflowNamespace      string
	KubeflowRuntime        string // ClusterTrainingRuntime name
	TrainingImage          string // image for Katib trials
	KubeflowTrainingSecret string // S3 credentials secret mounted into Katib trials
	HPOAlgorithm           string
	HPOMaxTrials           int
	HPOParallelTrials      int
	HPOTrialGradientSteps  int // offline updates per Katib trial
	HPOEvalEpisodes        int // test-track episodes per Katib trial (the objective)
}

func Load() *Config {
	return &Config{
		ServiceName:               getEnv("SERVICE_NAME", "fleetos"),
		Environment:               getEnv("ENVIRONMENT", "dev"),
		GRPCPort:                  getEnvInt("GRPC_PORT", 50051),
		HTTPPort:                  getEnvInt("HTTP_PORT", 8080),
		KafkaBrokers:              []string{getEnv("KAFKA_BROKERS", "localhost:9092")},
		KafkaTelemetryTopic:       getEnv("KAFKA_TELEMETRY_TOPIC", "robot.telemetry"),
		KafkaCommandTopic:         getEnv("KAFKA_COMMAND_TOPIC", "robot.commands"),
		KafkaCommandDispatchTopic: getEnv("KAFKA_COMMAND_DISPATCH_TOPIC", "robot.commands.dispatch"),
		KafkaCommandAcksTopic:     getEnv("KAFKA_COMMAND_ACKS_TOPIC", "robot.command-acks"),
		KafkaDeploymentEventTopic: getEnv("KAFKA_DEPLOYMENT_EVENT_TOPIC", "deployment.events"),
		KafkaGroupID:              getEnv("KAFKA_GROUP_ID", "fleetos-ingestion"),
		PostgresDSN:               getEnv("POSTGRES_DSN", "postgres://fleetos:fleetos@localhost:5432/fleetos?sslmode=disable"),
		RedisAddr:                 getEnv("REDIS_ADDR", "localhost:6379"),
		RedisPassword:             getEnv("REDIS_PASSWORD", ""),
		RedisDB:                   getEnvInt("REDIS_DB", 0),
		S3Endpoint:                getEnv("S3_ENDPOINT", "localhost:9000"),
		S3Bucket:                  getEnv("S3_BUCKET", "fleetos-telemetry"),
		S3AccessKey:               getEnv("S3_ACCESS_KEY", "fleetos"),
		S3SecretKey:               getEnv("S3_SECRET_KEY", "fleetos123"),
		S3UseSSL:                  getEnv("S3_USE_SSL", "false") == "true",
		InferenceEndpoint:         getEnv("INFERENCE_ENDPOINT", "localhost:8081"),
		InferenceTimeout:          time.Duration(getEnvInt("INFERENCE_TIMEOUT_MS", 5000)) * time.Millisecond,
		JWTSecret:                 getEnv("JWT_SECRET", "dev-secret-change-me"),
		OAuth2Issuer:              getEnv("OAUTH2_ISSUER", "https://auth.fleetos.dev"),
		RateLimitRPS:              getEnvInt("RATE_LIMIT_RPS", 100),
		RateLimitBurst:            getEnvInt("RATE_LIMIT_BURST", 200),
		TemporalHostPort:          getEnv("TEMPORAL_HOST_PORT", "localhost:7233"),
		TemporalNamespace:         getEnv("TEMPORAL_NAMESPACE", "default"),
		TemporalEnabled:           getEnv("TEMPORAL_ENABLED", "false") == "true",
		ArgoServerURL:             getEnv("ARGO_SERVER_URL", ""),
		ArgoNamespace:             getEnv("ARGO_NAMESPACE", "argo"),
		ArgoToken:                 getEnv("ARGO_TOKEN", ""),
		ArgoInsecureSkipVerify:    getEnv("ARGO_INSECURE_SKIP_VERIFY", "false") == "true",
		ModelsBucket:              getEnv("MODELS_BUCKET", "fleetos-models"),
		TelemetryBucket:           getEnv("TELEMETRY_BUCKET", "fleetos-telemetry"),
		MinEvalPassRate:           getEnvFloat("MODEL_MIN_EVAL_PASS_RATE", modelgate.DefaultMinPassRate),
		KubeflowEnabled:           getEnv("KUBEFLOW_ENABLED", "false") == "true",
		KubeflowNamespace:         getEnv("KUBEFLOW_NAMESPACE", "fleetos-training"),
		KubeflowRuntime:           getEnv("KUBEFLOW_RUNTIME", "fleetos-sb3"),
		TrainingImage:             getEnv("TRAINING_IMAGE", "fleetos/training:latest"),
		HPOAlgorithm:              getEnv("HPO_ALGORITHM", defaultHPOAlgorithm),
		HPOMaxTrials:              getEnvInt("HPO_MAX_TRIALS", defaultHPOMaxTrials),
		HPOParallelTrials:         getEnvInt("HPO_PARALLEL_TRIALS", defaultHPOParallelTrials),
		HPOTrialGradientSteps:     getEnvInt("HPO_TRIAL_GRADIENT_STEPS", defaultHPOTrialGradientSteps),
		HPOEvalEpisodes:           getEnvInt("HPO_EVAL_EPISODES", defaultHPOEvalEpisodes),
		KubeflowTrainingSecret:    getEnv("KUBEFLOW_TRAINING_SECRET", "fleetos-training"),
	}
}

// Katib defaults sized for CPU-only local clusters.
const (
	defaultHPOAlgorithm          = "tpe"
	defaultHPOMaxTrials          = 4
	defaultHPOParallelTrials     = 2
	defaultHPOTrialGradientSteps = 2_000
	defaultHPOEvalEpisodes       = 5
)

func getEnvFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			slog.Warn("invalid float config, using default", "key", key, "value", v, "default", fallback, "error", err)
			return fallback
		}
		return f
	}
	return fallback
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		i, err := strconv.Atoi(v)
		if err != nil {
			slog.Warn("invalid int config, using default", "key", key, "value", v, "default", fallback, "error", err)
			return fallback
		}
		return i
	}
	return fallback
}
