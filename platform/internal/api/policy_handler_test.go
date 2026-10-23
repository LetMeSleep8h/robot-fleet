package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/service"
)

type fakePolicyService struct {
	policy *service.RobotPolicy
	err    error
	gotID  string
}

func (f *fakePolicyService) GetRobotPolicy(_ context.Context, robotID string) (*service.RobotPolicy, error) {
	f.gotID = robotID
	return f.policy, f.err
}

func TestPolicyHandler_GetRobotPolicy(t *testing.T) {
	tests := []struct {
		name     string
		svc      *fakePolicyService
		wantCode int
	}{
		{"policy", &fakePolicyService{policy: &service.RobotPolicy{RobotID: "r1", ModelID: "fleet-v3", Version: "v3", ArtifactURL: "training/x/policy.zip", Source: service.PolicySourceFleetDefault}}, http.StatusOK},
		{"nothing deployed", &fakePolicyService{err: fmt.Errorf("x: %w", service.ErrNotFound)}, http.StatusNotFound},
		{"storage down", &fakePolicyService{err: fmt.Errorf("boom")}, http.StatusInternalServerError},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewPolicyHandler(tt.svc)
			req := httptest.NewRequest(http.MethodGet, "/api/v1/robots/r1/policy", nil)
			req.SetPathValue("id", "r1")
			rec := httptest.NewRecorder()

			h.GetRobotPolicy(rec, req)

			if rec.Code != tt.wantCode {
				t.Fatalf("code = %d, want %d (%s)", rec.Code, tt.wantCode, rec.Body.String())
			}
			if tt.svc.gotID != "r1" {
				t.Errorf("service called with %q", tt.svc.gotID)
			}
			if tt.wantCode == http.StatusOK {
				var got map[string]string
				if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
					t.Fatal(err)
				}
				for _, key := range []string{"robot_id", "model_id", "version", "artifact_url", "source"} {
					if got[key] == "" {
						t.Errorf("response missing %q: %v", key, got)
					}
				}
			}
		})
	}
}
