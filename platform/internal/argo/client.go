package argo

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

// Client talks to the Argo Workflows Server REST API. Safe for concurrent use.
type Client struct {
	baseURL   string
	namespace string
	token     string
	http      *http.Client
}

// NewClient creates a Client. A zero Timeout uses DefaultTimeout.
func NewClient(cfg Config) *Client {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if cfg.InsecureSkipVerify {
		transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // opt-in for local self-signed argo-server
	}
	return &Client{
		baseURL:   strings.TrimRight(cfg.ServerURL, "/"),
		namespace: cfg.Namespace,
		token:     cfg.Token,
		http:      &http.Client{Timeout: timeout, Transport: transport},
	}
}

// SubmitTemplate submits a new workflow from the named WorkflowTemplate and
// returns the generated workflow name.
func (c *Client) SubmitTemplate(ctx context.Context, template string, params, labels map[string]string) (string, error) {
	body := submitRequest{
		Namespace:    c.namespace,
		ResourceKind: "WorkflowTemplate",
		ResourceName: template,
		SubmitOptions: submitOptions{
			Parameters: joinPairs(params),
			Labels:     strings.Join(joinPairs(labels), ","),
		},
	}

	var wf workflowResponse
	path := fmt.Sprintf("/api/v1/workflows/%s/submit", url.PathEscape(c.namespace))
	if err := c.do(ctx, http.MethodPost, path, body, &wf); err != nil {
		return "", fmt.Errorf("submit workflow template %s: %w", template, err)
	}
	if wf.Metadata.Name == "" {
		return "", fmt.Errorf("submit workflow template %s: response has no workflow name", template)
	}
	return wf.Metadata.Name, nil
}

// GetWorkflowStatus returns the phase and message of a workflow. A workflow
// the controller has not picked up yet reports PhasePending.
func (c *Client) GetWorkflowStatus(ctx context.Context, name string) (Status, error) {
	var wf workflowResponse
	path := fmt.Sprintf("/api/v1/workflows/%s/%s?fields=status.phase,status.message",
		url.PathEscape(c.namespace), url.PathEscape(name))
	if err := c.do(ctx, http.MethodGet, path, nil, &wf); err != nil {
		return Status{}, fmt.Errorf("get workflow %s: %w", name, err)
	}
	phase := wf.Status.Phase
	if phase == "" {
		phase = PhasePending
	}
	return Status{Phase: phase, Message: wf.Status.Message}, nil
}

func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	var reqBody io.Reader
	if in != nil {
		data, err := json.Marshal(in)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		reqBody = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reqBody)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("call argo server: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck // read-only body

	if resp.StatusCode == http.StatusNotFound && method == http.MethodGet {
		return ErrWorkflowNotFound
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody)) // best-effort error context
		return fmt.Errorf("argo server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

// joinPairs renders a map as sorted "key=value" strings for deterministic requests.
func joinPairs(m map[string]string) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	pairs := make([]string, 0, len(keys))
	for _, k := range keys {
		pairs = append(pairs, k+"="+m[k])
	}
	return pairs
}
