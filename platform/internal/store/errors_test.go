package store

import (
	"errors"
	"fmt"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestNotFoundOr(t *testing.T) {
	other := errors.New("connection refused")
	tests := []struct {
		name         string
		err          error
		wantNotFound bool
	}{
		{"no rows", pgx.ErrNoRows, true},
		{"wrapped no rows", fmt.Errorf("scan: %w", pgx.ErrNoRows), true},
		{"invalid uuid text", &pgconn.PgError{Code: pgInvalidTextRepresentation}, true},
		{"other postgres error", &pgconn.PgError{Code: "23505"}, false},
		{"other error", other, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := notFoundOr(tt.err)
			if errors.Is(got, ErrNotFound) != tt.wantNotFound {
				t.Errorf("notFoundOr(%v) = %v, want not-found=%v", tt.err, got, tt.wantNotFound)
			}
			if !tt.wantNotFound && got != tt.err {
				t.Errorf("non-matching errors must pass through unchanged")
			}
		})
	}
	if notFoundOr(nil) != nil {
		t.Error("nil must stay nil")
	}
}

func TestRequireRowsAffected(t *testing.T) {
	if err := requireRowsAffected(pgconn.NewCommandTag("UPDATE 1")); err != nil {
		t.Errorf("1 row: %v", err)
	}
	if err := requireRowsAffected(pgconn.NewCommandTag("UPDATE 0")); !errors.Is(err, ErrNotFound) {
		t.Errorf("0 rows: want ErrNotFound, got %v", err)
	}
}
