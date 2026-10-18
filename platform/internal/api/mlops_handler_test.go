package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/auth"
	"github.com/dimuthu/robot-fleet/internal/modelgate"
	"github.com/dimuthu/robot-fleet/internal/service"
	"github.com/dimuthu/robot-fleet/internal/store"
)

// fakeModelService returns a scripted DeployModel result.
type fakeModelService struct {
	service.ModelRegistryService
	deployRes *service.DeployResult
	deployErr error
}

func (f *fakeModelService) DeployModel(context.Context, string) (*service.DeployResult, error) {
	return f.deployRes, f.deployErr
}

func TestModelHandler_DeployModel(t *testing.T) {
	tests := []struct {
		name     string
		res      *service.DeployResult
		err      error
		wantCode int
		wantBody string
	}{
		{"canary started", &service.DeployResult{ModelID: "m1", Status: service.DeployStatusCanaryStarted, WorkflowID: "deploy-m1"}, nil, http.StatusAccepted, "deploy-m1"},
		{"not found", nil, fmt.Errorf("get model m1: %w", service.ErrNotFound), http.StatusNotFound, ""},
		{"not staged", nil, fmt.Errorf("x: %w", service.ErrConflict), http.StatusConflict, ""},
		{"gate rejected", nil, fmt.Errorf("deploy model m1: %w: eval_pass_rate 0.00 is below the minimum 0.50", modelgate.ErrRejected), http.StatusUnprocessableEntity, "eval_pass_rate 0.00"},
		{"no rollout", nil, fmt.Errorf("x: %w", service.ErrUnavailable), http.StatusServiceUnavailable, ""},
		{"internal", nil, fmt.Errorf("boom"), http.StatusInternalServerError, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewModelHandler(&fakeModelService{deployRes: tt.res, deployErr: tt.err})
			req := httptest.NewRequest(http.MethodPost, "/api/v1/models/m1/deploy", nil)
			req.SetPathValue("id", "m1")
			rec := httptest.NewRecorder()

			h.DeployModel(rec, req)

			if rec.Code != tt.wantCode {
				t.Errorf("code = %d, want %d (body %s)", rec.Code, tt.wantCode, rec.Body.String())
			}
			if tt.wantBody != "" && !strings.Contains(rec.Body.String(), tt.wantBody) {
				t.Errorf("body %q missing %q", rec.Body.String(), tt.wantBody)
			}
		})
	}
}

// fakeTrainingService returns scripted errors.
type fakeTrainingService struct {
	service.TrainingService
	submitErr   error
	callbackErr error
	evalErr     error
}

func (f *fakeTrainingService) SubmitJob(context.Context, string, string, service.SubmitTrainingJobRequest) (*store.TrainingJobRecord, error) {
	if f.submitErr != nil {
		return nil, f.submitErr
	}
	return &store.TrainingJobRecord{ID: "j1", Status: service.TrainingStatusRunning}, nil
}

func (f *fakeTrainingService) SubmitEval(context.Context, string, service.SubmitEvalRequest) (*store.TrainingEvalRecord, error) {
	if f.evalErr != nil {
		return nil, f.evalErr
	}
	return &store.TrainingEvalRecord{ID: "e1"}, nil
}

func (f *fakeTrainingService) HandleTrainingCallback(context.Context, service.TrainingCallbackPayload) error {
	return f.callbackErr
}

func (f *fakeTrainingService) HandleEvalCallback(context.Context, service.EvalCallbackPayload) error {
	return f.callbackErr
}

func withTenant(r *http.Request) *http.Request {
	ctx := context.WithValue(r.Context(), auth.TenantIDKey, "tenant-dev")
	return r.WithContext(context.WithValue(ctx, auth.RoleKey, auth.RoleAdmin))
}

func TestTrainingHandler_SubmitJobErrors(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantCode int
	}{
		{"accepted", nil, http.StatusAccepted},
		{"training not configured", fmt.Errorf("x: %w", service.ErrUnavailable), http.StatusServiceUnavailable},
		{"unsupported algorithm", fmt.Errorf("x: %w", service.ErrInvalidInput), http.StatusBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewTrainingHandler(&fakeTrainingService{submitErr: tt.err})
			req := withTenant(httptest.NewRequest(http.MethodPost, "/api/v1/locomotion/jobs", strings.NewReader(`{"timesteps":20000}`)))
			rec := httptest.NewRecorder()

			h.SubmitJob(rec, req)

			if rec.Code != tt.wantCode {
				t.Errorf("code = %d, want %d (body %s)", rec.Code, tt.wantCode, rec.Body.String())
			}
		})
	}
}

func TestTrainingHandler_SubmitEvalConflict(t *testing.T) {
	h := NewTrainingHandler(&fakeTrainingService{evalErr: fmt.Errorf("job running: %w", service.ErrConflict)})
	req := withTenant(httptest.NewRequest(http.MethodPost, "/api/v1/locomotion/evals", strings.NewReader(`{"job_id":"j1"}`)))
	rec := httptest.NewRecorder()

	h.SubmitEval(rec, req)

	if rec.Code != http.StatusConflict {
		t.Errorf("code = %d, want 409", rec.Code)
	}
}

func TestTrainingHandler_CallbacksForUnknownIDsAre404(t *testing.T) {
	notFound := fmt.Errorf("complete training job loco-k3s-001: %w", service.ErrNotFound)
	h := NewTrainingHandler(&fakeTrainingService{callbackErr: notFound})

	for name, call := range map[string]func(http.ResponseWriter, *http.Request){
		"training": h.TrainingCallback,
		"eval":     h.EvalCallback,
	} {
		t.Run(name, func(t *testing.T) {
			body, _ := json.Marshal(map[string]any{"job_id": "loco-k3s-001", "eval_id": "loco-k3s-001", "status": "completed"})
			rec := httptest.NewRecorder()
			call(rec, httptest.NewRequest(http.MethodPost, "/cb", strings.NewReader(string(body))))
			if rec.Code != http.StatusNotFound {
				t.Errorf("code = %d, want 404", rec.Code)
			}
		})
	}
}
