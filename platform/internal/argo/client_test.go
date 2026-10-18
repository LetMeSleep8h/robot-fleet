package argo

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func newTestClient(t *testing.T, h http.HandlerFunc, token string) *Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return NewClient(Config{ServerURL: srv.URL, Namespace: "argo", Token: token, Timeout: 5 * time.Second})
}

func TestSubmitTemplate_SendsSubmitRequest(t *testing.T) {
	var got submitRequest
	var gotPath, gotAuth string
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.Method + " " + r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode body: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"metadata": map[string]any{"name": "fleetos-locomotion-training-x7k2p"}})
	}, "s3cret")

	name, err := c.SubmitTemplate(context.Background(), "fleetos-locomotion-training",
		map[string]string{"timesteps": "20000", "job-id": "job-1"},
		map[string]string{"fleetos.io/job-id": "job-1"})
	if err != nil {
		t.Fatalf("SubmitTemplate: %v", err)
	}
	if name != "fleetos-locomotion-training-x7k2p" {
		t.Errorf("name = %q", name)
	}
	if gotPath != "POST /api/v1/workflows/argo/submit" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer s3cret" {
		t.Errorf("auth = %q", gotAuth)
	}
	if got.ResourceKind != "WorkflowTemplate" || got.ResourceName != "fleetos-locomotion-training" || got.Namespace != "argo" {
		t.Errorf("unexpected request: %+v", got)
	}
	// Parameters are sorted by key so submissions are deterministic.
	want := []string{"job-id=job-1", "timesteps=20000"}
	if len(got.SubmitOptions.Parameters) != 2 || got.SubmitOptions.Parameters[0] != want[0] || got.SubmitOptions.Parameters[1] != want[1] {
		t.Errorf("parameters = %v, want %v", got.SubmitOptions.Parameters, want)
	}
	if got.SubmitOptions.Labels != "fleetos.io/job-id=job-1" {
		t.Errorf("labels = %q", got.SubmitOptions.Labels)
	}
}

func TestSubmitTemplate_NoTokenOmitsAuthHeader(t *testing.T) {
	var gotAuth string
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewEncoder(w).Encode(map[string]any{"metadata": map[string]any{"name": "wf"}})
	}, "")

	if _, err := c.SubmitTemplate(context.Background(), "t", nil, nil); err != nil {
		t.Fatalf("SubmitTemplate: %v", err)
	}
	if gotAuth != "" {
		t.Errorf("expected no Authorization header, got %q", gotAuth)
	}
}

func TestSubmitTemplate_ServerError(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"message":"workflowtemplates.argoproj.io \"nope\" not found"}`, http.StatusNotFound)
	}, "")

	_, err := c.SubmitTemplate(context.Background(), "nope", nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestSubmitTemplate_EmptyNameIsError(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"metadata": map[string]any{}})
	}, "")

	if _, err := c.SubmitTemplate(context.Background(), "t", nil, nil); err == nil {
		t.Fatal("expected error for response without workflow name")
	}
}

func TestGetWorkflowStatus(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		code    int
		want    Status
		wantErr error
	}{
		{"succeeded", `{"status":{"phase":"Succeeded"}}`, 200, Status{Phase: PhaseSucceeded}, nil},
		{"failed with message", `{"status":{"phase":"Failed","message":"child failed"}}`, 200, Status{Phase: PhaseFailed, Message: "child failed"}, nil},
		{"not started yet", `{"status":{}}`, 200, Status{Phase: PhasePending}, nil},
		{"not found", `{"message":"not found"}`, 404, Status{}, ErrWorkflowNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotPath string
			c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				w.WriteHeader(tt.code)
				_, _ = w.Write([]byte(tt.body))
			}, "")

			got, err := c.GetWorkflowStatus(context.Background(), "wf-1")
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("err = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("GetWorkflowStatus: %v", err)
			}
			if got != tt.want {
				t.Errorf("status = %+v, want %+v", got, tt.want)
			}
			if gotPath != "/api/v1/workflows/argo/wf-1" {
				t.Errorf("path = %q", gotPath)
			}
		})
	}
}

func TestStatus_Done(t *testing.T) {
	for phase, done := range map[string]bool{
		PhasePending: false, PhaseRunning: false,
		PhaseSucceeded: true, PhaseFailed: true, PhaseError: true,
	} {
		if got := (Status{Phase: phase}).Done(); got != done {
			t.Errorf("Status{%s}.Done() = %v, want %v", phase, got, done)
		}
	}
}

func TestGetWorkflowStatus_RespectsContextCancel(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}, "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.GetWorkflowStatus(ctx, "wf"); err == nil {
		t.Fatal("expected error from cancelled context")
	}
}
