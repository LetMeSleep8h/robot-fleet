// Package argo is a minimal client for the Argo Workflows Server REST API.
//
// Services submit training/evaluation workflows from pre-installed
// WorkflowTemplates and poll their status over HTTP, so they need no kubectl
// binary or Kubernetes client inside the container.
package argo

import (
	"errors"
	"time"
)

// Workflow phases reported by Argo (status.phase).
const (
	PhasePending   = "Pending"
	PhaseRunning   = "Running"
	PhaseSucceeded = "Succeeded"
	PhaseFailed    = "Failed"
	PhaseError     = "Error"
)

// DefaultTimeout bounds every request to the Argo Server.
const DefaultTimeout = 15 * time.Second

// maxErrorBody caps how much of an error response is echoed into errors.
const maxErrorBody = 512

// ErrWorkflowNotFound is returned when the requested workflow does not exist.
var ErrWorkflowNotFound = errors.New("argo workflow not found")

// Config configures a Client.
type Config struct {
	ServerURL          string // e.g. https://argo-server.argo.svc.cluster.local:2746
	Namespace          string // namespace holding the WorkflowTemplates, e.g. "argo"
	Token              string // optional bearer token (empty when argo-server runs --auth-mode=server)
	InsecureSkipVerify bool   // accept argo-server's self-signed certificate (local clusters only)
	Timeout            time.Duration
}

// Status is the current state of a workflow.
type Status struct {
	Phase   string
	Message string
}

// Done reports whether the workflow has reached a terminal phase.
func (s Status) Done() bool {
	return s.Phase == PhaseSucceeded || s.Phase == PhaseFailed || s.Phase == PhaseError
}

// submitRequest is the body of POST /api/v1/workflows/{namespace}/submit.
type submitRequest struct {
	Namespace     string        `json:"namespace"`
	ResourceKind  string        `json:"resourceKind"`
	ResourceName  string        `json:"resourceName"`
	SubmitOptions submitOptions `json:"submitOptions"`
}

type submitOptions struct {
	Parameters []string `json:"parameters,omitempty"`
	Labels     string   `json:"labels,omitempty"`
}

// workflowResponse captures the fields we read from a Workflow object.
type workflowResponse struct {
	Metadata struct {
		Name string `json:"name"`
	} `json:"metadata"`
	Status struct {
		Phase   string `json:"phase"`
		Message string `json:"message"`
	} `json:"status"`
}
