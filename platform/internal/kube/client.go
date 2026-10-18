package kube

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
)

// Client talks to the Kubernetes API server. Safe for concurrent use.
type Client struct {
	host      string
	tokenFile string
	http      *http.Client
}

// InClusterConfig returns the configuration for a pod's own service account.
func InClusterConfig() (Config, error) {
	host, port := os.Getenv("KUBERNETES_SERVICE_HOST"), os.Getenv("KUBERNETES_SERVICE_PORT")
	if host == "" || port == "" {
		return Config{}, fmt.Errorf("not running in a cluster: KUBERNETES_SERVICE_HOST/PORT unset")
	}
	return Config{
		Host:      "https://" + net.JoinHostPort(host, port),
		TokenFile: serviceAccountDir + "/token",
		CAFile:    serviceAccountDir + "/ca.crt",
	}, nil
}

// NewClient creates a Client. A zero Timeout uses DefaultTimeout.
func NewClient(cfg Config) (*Client, error) {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if cfg.CAFile != "" {
		pem, err := os.ReadFile(cfg.CAFile)
		if err != nil {
			return nil, fmt.Errorf("read CA file: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("no certificates in CA file %s", cfg.CAFile)
		}
		transport.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
	}
	return &Client{
		host:      strings.TrimRight(cfg.Host, "/"),
		tokenFile: cfg.TokenFile,
		http:      &http.Client{Timeout: timeout, Transport: transport},
	}, nil
}

// Create creates a namespaced object and returns the stored object.
func (c *Client) Create(ctx context.Context, r Resource, namespace string, obj map[string]any) (map[string]any, error) {
	var out map[string]any
	if err := c.do(ctx, http.MethodPost, collectionPath(r, namespace), obj, &out); err != nil {
		return nil, fmt.Errorf("create %s in %s: %w", r.Resource, namespace, err)
	}
	return out, nil
}

// Get reads a namespaced object.
func (c *Client) Get(ctx context.Context, r Resource, namespace, name string) (map[string]any, error) {
	var out map[string]any
	path := collectionPath(r, namespace) + "/" + url.PathEscape(name)
	if err := c.do(ctx, http.MethodGet, path, nil, &out); err != nil {
		return nil, fmt.Errorf("get %s %s/%s: %w", r.Resource, namespace, name, err)
	}
	return out, nil
}

func collectionPath(r Resource, namespace string) string {
	return fmt.Sprintf("/apis/%s/%s/namespaces/%s/%s", r.Group, r.Version, url.PathEscape(namespace), r.Resource)
}

func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	var body io.Reader
	if in != nil {
		data, err := json.Marshal(in)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		body = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.host+path, body)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.tokenFile != "" {
		token, err := os.ReadFile(c.tokenFile)
		if err != nil {
			return fmt.Errorf("read service account token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(string(token)))
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("call api server: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck // read-only body

	switch {
	case resp.StatusCode == http.StatusNotFound:
		return ErrNotFound
	case resp.StatusCode == http.StatusConflict:
		return ErrAlreadyExists
	case resp.StatusCode < 200 || resp.StatusCode >= 300:
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody)) // best-effort error context
		return fmt.Errorf("api server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

// Name returns metadata.name of an object, or "".
func Name(obj map[string]any) string {
	meta, _ := obj["metadata"].(map[string]any)
	name, _ := meta["name"].(string)
	return name
}

// Condition returns the status condition of the given type.
func Condition(obj map[string]any, condType string) (StatusCondition, bool) {
	status, _ := obj["status"].(map[string]any)
	conds, _ := status["conditions"].([]any)
	for _, raw := range conds {
		c, _ := raw.(map[string]any)
		if t, _ := c["type"].(string); t == condType {
			return StatusCondition{
				Type:    t,
				Status:  str(c["status"]),
				Reason:  str(c["reason"]),
				Message: str(c["message"]),
			}, true
		}
	}
	return StatusCondition{}, false
}

func str(v any) string {
	s, _ := v.(string)
	return s
}
