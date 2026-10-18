// Package modelgate decides whether a trained model is good enough to ship.
//
// The same policy guards both paths that can promote a model: the Temporal
// training pipeline (before registering/auto-deploying) and the registry's
// deploy endpoint (before starting a canary rollout).
package modelgate

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Metric keys written into model_registry.metrics by the evaluation step.
const (
	MetricEvalPassRate   = "eval_pass_rate"   // fraction of eval scenarios passed, 0..1
	MetricEvalMeanReward = "eval_mean_reward" // mean episode reward over eval scenarios
)

// Defaults used when no policy is configured.
const (
	DefaultMinPassRate         = 0.5  // at least half the eval scenarios must pass
	DefaultMaxRewardRegression = 0.05 // may be at most 5% worse than the baseline's reward
)

// ErrRejected is wrapped by every gate rejection.
var ErrRejected = errors.New("model rejected by evaluation gate")

// Policy holds the promotion thresholds.
type Policy struct {
	MinPassRate         float64
	MaxRewardRegression float64
}

// DefaultPolicy returns the default promotion thresholds.
func DefaultPolicy() Policy {
	return Policy{MinPassRate: DefaultMinPassRate, MaxRewardRegression: DefaultMaxRewardRegression}
}

// Check returns nil when the candidate's evaluation metrics clear the policy,
// or an error wrapping ErrRejected explaining why not. baseline may be nil
// (first model ever, or the baseline was never evaluated).
func (p Policy) Check(candidate, baseline map[string]any) error {
	passRate, ok := metric(candidate, MetricEvalPassRate)
	if !ok {
		return fmt.Errorf("%w: no %s metric (model was never evaluated)", ErrRejected, MetricEvalPassRate)
	}
	if passRate < p.MinPassRate {
		return fmt.Errorf("%w: %s %.2f is below the minimum %.2f", ErrRejected, MetricEvalPassRate, passRate, p.MinPassRate)
	}

	baseReward, hasBaseline := metric(baseline, MetricEvalMeanReward)
	if !hasBaseline || baseReward <= 0 {
		return nil
	}
	reward, ok := metric(candidate, MetricEvalMeanReward)
	if !ok {
		return fmt.Errorf("%w: no %s metric to compare against baseline %.2f", ErrRejected, MetricEvalMeanReward, baseReward)
	}
	if floor := baseReward * (1 - p.MaxRewardRegression); reward < floor {
		return fmt.Errorf("%w: %s %.2f regresses past %.2f (baseline %.2f)", ErrRejected, MetricEvalMeanReward, reward, floor, baseReward)
	}
	return nil
}

// metric reads a numeric metric that may have been decoded from JSON.
func metric(m map[string]any, key string) (float64, bool) {
	switch v := m[key].(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
