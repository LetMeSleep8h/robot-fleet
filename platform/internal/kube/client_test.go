package kube

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var testResource = Resource{Group: "trainer.kubeflow.org", Version: "v1alpha1", Resource: "trainjobs"}

func newTestClient(t *testing.T, h http.HandlerFunc, tokenFile string) *Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	c, err := NewClient(Config{Host: srv.URL, TokenFile: tokenFile, Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return c
}

func TestCreate_PostsToNamespacedResourcePath(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	tokenFile := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenFile, []byte("tok-1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.Method + " " + r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]any{"metadata": map[string]any{"name": "job-1"}})
	}, tokenFile)

	out, err := c.Create(context.Background(), testResource, "fleetos-training", map[string]any{"kind": "TrainJob"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if gotPath != "POST /apis/trainer.kubeflow.org/v1alpha1/namespaces/fleetos-training/trainjobs" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer tok-1" {
		t.Errorf("auth = %q (token file must be trimmed)", gotAuth)
	}
	if gotBody["kind"] != "TrainJob" {
		t.Errorf("body = %v", gotBody)
	}
	if Name(out) != "job-1" {
		t.Errorf("Name(out) = %q", Name(out))
	}
}

func TestToken_IsReReadEachRequest(t *testing.T) {
	tokenFile := filepath.Join(t.TempDir(), "token")
	_ = os.WriteFile(tokenFile, []byte("old"), 0o600)
	var auths []string
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		auths = append(auths, r.Header.Get("Authorization"))
		_, _ = w.Write([]byte(`{}`))
	}, tokenFile)

	_, _ = c.Get(context.Background(), testResource, "ns", "a")
	_ = os.WriteFile(tokenFile, []byte("rotated"), 0o600) // kubelet rotates projected SA tokens
	_, _ = c.Get(context.Background(), testResource, "ns", "a")

	if len(auths) != 2 || auths[1] != "Bearer rotated" {
		t.Errorf("rotated token not picked up: %v", auths)
	}
}

func TestGet_NotFound(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"reason":"NotFound"}`, http.StatusNotFound)
	}, "")
	_, err := c.Get(context.Background(), testResource, "ns", "missing")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestCreate_Conflict(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"reason":"AlreadyExists"}`, http.StatusConflict)
	}, "")
	_, err := c.Create(context.Background(), testResource, "ns", map[string]any{})
	if !errors.Is(err, ErrAlreadyExists) {
		t.Fatalf("err = %v, want ErrAlreadyExists", err)
	}
}

func TestCreate_ServerErrorIncludesBody(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `admission webhook denied: runtimeRef not found`, http.StatusUnprocessableEntity)
	}, "")
	_, err := c.Create(context.Background(), testResource, "ns", map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "runtimeRef not found") {
		t.Fatalf("err = %v", err)
	}
}

func TestInClusterConfig(t *testing.T) {
	t.Setenv("KUBERNETES_SERVICE_HOST", "10.43.0.1")
	t.Setenv("KUBERNETES_SERVICE_PORT", "443")
	cfg, err := InClusterConfig()
	if err != nil {
		t.Fatalf("InClusterConfig: %v", err)
	}
	if cfg.Host != "https://10.43.0.1:443" || cfg.TokenFile != serviceAccountDir+"/token" || cfg.CAFile != serviceAccountDir+"/ca.crt" {
		t.Errorf("unexpected config %+v", cfg)
	}

	t.Setenv("KUBERNETES_SERVICE_HOST", "")
	if _, err := InClusterConfig(); err == nil {
		t.Error("expected error outside a cluster")
	}
}

func TestNestedHelpers(t *testing.T) {
	obj := map[string]any{"status": map[string]any{
		"conditions": []any{
			map[string]any{"type": "Created", "status": "True"},
			map[string]any{"type": "Complete", "status": "True", "message": "done"},
		},
	}}
	if c, ok := Condition(obj, "Complete"); !ok || c.Status != "True" || c.Message != "done" {
		t.Errorf("Condition = %+v %v", c, ok)
	}
	if _, ok := Condition(obj, "Failed"); ok {
		t.Error("absent condition must report false")
	}
	if Name(map[string]any{}) != "" {
		t.Error("Name of empty object must be empty")
	}
}
