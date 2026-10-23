package middleware

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dimuthu/robot-fleet/internal/auth"
)

// exhaustedQuota reports a usage counter far above every tier's daily limit.
type exhaustedQuota struct{ mockRedis }

func (e *exhaustedQuota) GetUsageCounter(context.Context, string, string, string) (int64, error) {
	return 1_000_000, nil
}

func tenantRequest(method, path string) *http.Request {
	req := httptest.NewRequest(method, path, nil)
	return req.WithContext(context.WithValue(req.Context(), auth.TenantIDKey, "tenant-demo"))
}

func TestIsControlPlane(t *testing.T) {
	tests := []struct {
		method, path string
		want         bool
	}{
		{http.MethodGet, "/api/v1/robots/robot-0001/policy", true},
		{http.MethodPost, "/api/v1/robots/robot-0001/policy", false},
		{http.MethodGet, "/api/v1/robots/robot-0001", false},
		{http.MethodGet, "/api/v1/robots/robot-0001/policy/extra", false},
		{http.MethodGet, "/api/v1/robots//policy", false},
		{http.MethodGet, "/api/v1/models", false},
	}
	for _, tt := range tests {
		if got := IsControlPlane(httptest.NewRequest(tt.method, tt.path, nil)); got != tt.want {
			t.Errorf("IsControlPlane(%s %s) = %v, want %v", tt.method, tt.path, got, tt.want)
		}
	}
}

func TestQuotaEnforcement_ControlPlaneIsNeverBlocked(t *testing.T) {
	// Robots must keep receiving their policy even when the tenant's
	// developer-API quota is exhausted.
	handler := QuotaEnforcement(&exhaustedQuota{}, func(string) string { return "free" })(
		http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, tenantRequest(http.MethodGet, "/api/v1/robots/robot-0001/policy"))
	if rec.Code != http.StatusOK {
		t.Errorf("control-plane call blocked by quota: %d", rec.Code)
	}

	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, tenantRequest(http.MethodGet, "/api/v1/models"))
	if rec.Code != http.StatusPaymentRequired {
		t.Errorf("billable call should still hit the quota, got %d", rec.Code)
	}
}

func TestUsageMetering_ControlPlaneIsNotBilled(t *testing.T) {
	mock := &mockRedis{counters: map[string]int64{}}
	handler := UsageMetering(mock)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))

	handler.ServeHTTP(httptest.NewRecorder(), tenantRequest(http.MethodGet, "/api/v1/robots/robot-0001/policy"))
	handler.ServeHTTP(httptest.NewRecorder(), tenantRequest(http.MethodGet, "/api/v1/robots"))

	if got := mock.counters["tenant-demo:api_calls"]; got != 1 {
		t.Errorf("api_calls = %d, want 1 (only the billable call)", got)
	}
}
