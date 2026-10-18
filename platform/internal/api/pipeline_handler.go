package api

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/dimuthu/robot-fleet/internal/auth"
	"github.com/dimuthu/robot-fleet/internal/service"
)

// defaultPipelineTask is used when a pipeline request omits task_type.
const defaultPipelineTask = "locomotion"

// PipelineHandler exposes the Temporal training pipeline over HTTP.
type PipelineHandler struct {
	svc service.PipelineService
}

// NewPipelineHandler creates a PipelineHandler.
func NewPipelineHandler(svc service.PipelineService) *PipelineHandler {
	return &PipelineHandler{svc: svc}
}

// StartPipeline handles POST /api/v1/training/pipelines.
func (h *PipelineHandler) StartPipeline(w http.ResponseWriter, r *http.Request) {
	var req service.PipelineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.TaskType == "" {
		req.TaskType = defaultPipelineTask
	}

	tenantID := auth.GetTenantID(r.Context())
	run, err := h.svc.StartPipeline(r.Context(), tenantID, req)
	if err != nil {
		slog.Error("start training pipeline failed", "tenant_id", tenantID, "error", err)
		writeServiceError(w, err, "failed to start training pipeline")
		return
	}
	writeJSON(w, http.StatusAccepted, run)
}

// GetPipeline handles GET /api/v1/training/pipelines/{id}.
func (h *PipelineHandler) GetPipeline(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	run, err := h.svc.GetPipeline(r.Context(), id)
	if err != nil {
		slog.Error("get training pipeline failed", "pipeline", id, "error", err)
		writeServiceError(w, err, "failed to get training pipeline")
		return
	}
	writeJSON(w, http.StatusOK, run)
}
