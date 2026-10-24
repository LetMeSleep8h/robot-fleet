package middleware

import (
	"net/http"
	"regexp"
)

// controlPlaneRoutes are machine-to-machine calls robots make to keep
// operating (which policy to run). They stay authenticated and rate-limited
// but are neither billed nor blocked by a tenant's developer-API quota: a
// robot must not lose its model because the quota ran out.
var controlPlaneRoutes = []struct {
	method string
	path   *regexp.Regexp
}{
	{http.MethodGet, regexp.MustCompile(`^/api/v1/robots/[^/]+/policy$`)},
}

// IsControlPlane reports whether r is a robot control-plane call.
func IsControlPlane(r *http.Request) bool {
	for _, route := range controlPlaneRoutes {
		if r.Method == route.method && route.path.MatchString(r.URL.Path) {
			return true
		}
	}
	return false
}
