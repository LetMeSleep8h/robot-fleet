package api

import (
	"errors"
	"net/http"

	"github.com/dimuthu/robot-fleet/internal/modelgate"
	"github.com/dimuthu/robot-fleet/internal/service"
)

// statusForError maps service sentinel errors to HTTP status codes.
func statusForError(err error) int {
	switch {
	case errors.Is(err, service.ErrNotFound):
		return http.StatusNotFound
	case errors.Is(err, service.ErrInvalidInput):
		return http.StatusBadRequest
	case errors.Is(err, service.ErrConflict):
		return http.StatusConflict
	case errors.Is(err, modelgate.ErrRejected):
		return http.StatusUnprocessableEntity
	case errors.Is(err, service.ErrUnavailable):
		return http.StatusServiceUnavailable
	default:
		return http.StatusInternalServerError
	}
}

// writeServiceError writes err with its mapped status. Client errors carry the
// service's explanation (e.g. why the evaluation gate rejected a model);
// server errors get the generic fallback message.
func writeServiceError(w http.ResponseWriter, err error, fallback string) {
	status := statusForError(err)
	msg := fallback
	if status < http.StatusInternalServerError || status == http.StatusServiceUnavailable {
		msg = err.Error()
	}
	writeError(w, status, msg)
}
