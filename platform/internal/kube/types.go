// Package kube is a minimal Kubernetes REST client for creating and reading
// custom resources (Kubeflow TrainJobs, Katib Experiments) from inside the
// cluster, without the client-go dependency tree.
package kube

import (
	"errors"
	"time"
)

// DefaultTimeout bounds every request to the API server.
const DefaultTimeout = 15 * time.Second

// serviceAccountDir is where Kubernetes mounts the pod's service account.
const serviceAccountDir = "/var/run/secrets/kubernetes.io/serviceaccount"

// maxErrorBody caps how much of an error response is echoed into errors.
const maxErrorBody = 512

// Sentinel errors.
var (
	ErrNotFound      = errors.New("kubernetes object not found")
	ErrAlreadyExists = errors.New("kubernetes object already exists")
)

// Config configures a Client.
type Config struct {
	Host      string // e.g. https://10.43.0.1:443
	TokenFile string // re-read per request (projected tokens rotate)
	CAFile    string // API server CA; empty = system roots
	Timeout   time.Duration
}

// Resource identifies a namespaced API resource.
type Resource struct {
	Group    string
	Version  string
	Resource string // plural, e.g. "trainjobs"
}

// StatusCondition is one entry of status.conditions.
type StatusCondition struct {
	Type    string
	Status  string
	Reason  string
	Message string
}
