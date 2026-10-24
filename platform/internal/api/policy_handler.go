package api

import (
	"log/slog"
	"net/http"

	"github.com/dimuthu/robot-fleet/internal/service"
)

// PolicyHandler tells robots which policy to execute locally.
type PolicyHandler struct {
	svc service.PolicyService
}

// NewPolicyHandler creates a PolicyHandler.
func NewPolicyHandler(svc service.PolicyService) *PolicyHandler {
	return &PolicyHandler{svc: svc}
}

// GetRobotPolicy handles GET /api/v1/robots/{id}/policy.
func (h *PolicyHandler) GetRobotPolicy(w http.ResponseWriter, r *http.Request) {
	robotID := r.PathValue("id")
	policy, err := h.svc.GetRobotPolicy(r.Context(), robotID)
	if err != nil {
		if statusForError(err) >= http.StatusInternalServerError {
			slog.Error("get robot policy failed", "robot_id", robotID, "error", err)
		}
		writeServiceError(w, err, "failed to get robot policy")
		return
	}
	writeJSON(w, http.StatusOK, policy)
}
