package modelgate

import (
	"errors"
	"testing"
)

func TestPolicyCheck(t *testing.T) {
	p := Policy{MinPassRate: 0.5, MaxRewardRegression: 0.05}

	tests := []struct {
		name      string
		candidate map[string]any
		baseline  map[string]any
		wantPass  bool
	}{
		{"no evaluation results", map[string]any{"eval_mean_reward": 100.0}, nil, false},
		{"nil metrics", nil, nil, false},
		{"below pass rate", map[string]any{"eval_pass_rate": 0.0, "eval_mean_reward": 45.6}, nil, false},
		{"meets pass rate, no baseline", map[string]any{"eval_pass_rate": 0.5}, nil, true},
		{"int pass rate accepted", map[string]any{"eval_pass_rate": 1}, nil, true},
		{"within regression budget", map[string]any{"eval_pass_rate": 0.9, "eval_mean_reward": 96.0},
			map[string]any{"eval_mean_reward": 100.0}, true},
		{"regresses past budget", map[string]any{"eval_pass_rate": 0.9, "eval_mean_reward": 90.0},
			map[string]any{"eval_mean_reward": 100.0}, false},
		{"baseline without reward ignored", map[string]any{"eval_pass_rate": 0.9, "eval_mean_reward": 1.0},
			map[string]any{"success_rate": 0.99}, true},
		{"candidate missing reward vs baseline", map[string]any{"eval_pass_rate": 0.9},
			map[string]any{"eval_mean_reward": 100.0}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := p.Check(tt.candidate, tt.baseline)
			if tt.wantPass && err != nil {
				t.Fatalf("expected pass, got %v", err)
			}
			if !tt.wantPass {
				if err == nil {
					t.Fatal("expected rejection")
				}
				if !errors.Is(err, ErrRejected) {
					t.Fatalf("expected ErrRejected, got %v", err)
				}
			}
		})
	}
}

func TestDefaultPolicy(t *testing.T) {
	p := DefaultPolicy()
	if p.MinPassRate != DefaultMinPassRate || p.MaxRewardRegression != DefaultMaxRewardRegression {
		t.Errorf("unexpected default policy: %+v", p)
	}
}
