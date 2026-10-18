package api

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/service"
)

type fakePipelineService struct {
	gotReq   service.PipelineRequest
	startErr error
	getRes   *service.PipelineRun
	getErr   error
}

func (f *fakePipelineService) StartPipeline(_ context.Context, _ string, req service.PipelineRequest) (*service.PipelineRun, error) {
	f.gotReq = req
	if f.startErr != nil {
		return nil, f.startErr
	}
	return &service.PipelineRun{WorkflowID: "pipeline-pipe-1", JobID: "pipe-1", Status: service.PipelineStatusRunning}, nil
}

func (f *fakePipelineService) GetPipeline(context.Context, string) (*service.PipelineRun, error) {
	return f.getRes, f.getErr
}

func TestPipelineHandler_Start(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		err      error
		wantCode int
	}{
		{"started", `{"task_type":"locomotion","backend":"kubeflow","use_katib":true}`, nil, http.StatusAccepted},
		{"defaults task type", `{}`, nil, http.StatusAccepted},
		{"bad json", `{`, nil, http.StatusBadRequest},
		{"invalid", `{"task_type":"grasp"}`, fmt.Errorf("x: %w", service.ErrInvalidInput), http.StatusBadRequest},
		{"no temporal", `{"task_type":"locomotion"}`, fmt.Errorf("x: %w", service.ErrUnavailable), http.StatusServiceUnavailable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &fakePipelineService{startErr: tt.err}
			h := NewPipelineHandler(svc)
			rec := httptest.NewRecorder()
			h.StartPipeline(rec, withTenant(httptest.NewRequest(http.MethodPost, "/api/v1/training/pipelines", strings.NewReader(tt.body))))

			if rec.Code != tt.wantCode {
				t.Errorf("code = %d, want %d (%s)", rec.Code, tt.wantCode, rec.Body.String())
			}
			if tt.name == "defaults task type" && svc.gotReq.TaskType != "locomotion" {
				t.Errorf("task_type default = %q", svc.gotReq.TaskType)
			}
			if tt.wantCode == http.StatusAccepted && !strings.Contains(rec.Body.String(), "pipeline-pipe-1") {
				t.Errorf("body = %s", rec.Body.String())
			}
		})
	}
}

func TestPipelineHandler_Get(t *testing.T) {
	h := NewPipelineHandler(&fakePipelineService{getErr: fmt.Errorf("x: %w", service.ErrNotFound)})
	req := httptest.NewRequest(http.MethodGet, "/api/v1/training/pipelines/nope", nil)
	req.SetPathValue("id", "nope")
	rec := httptest.NewRecorder()
	h.GetPipeline(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Errorf("code = %d, want 404", rec.Code)
	}

	h = NewPipelineHandler(&fakePipelineService{getRes: &service.PipelineRun{WorkflowID: "pipeline-1", Status: service.PipelineStatusCompleted}})
	rec = httptest.NewRecorder()
	h.GetPipeline(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"completed"`) {
		t.Errorf("code = %d body = %s", rec.Code, rec.Body.String())
	}
}
