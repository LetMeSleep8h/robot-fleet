package store

import (
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// ErrNotFound is returned when a requested record does not exist.
var ErrNotFound = errors.New("not found")

// pgInvalidTextRepresentation is raised when an ID isn't a valid UUID; for a
// lookup by ID that simply means "no such record".
const pgInvalidTextRepresentation = "22P02"

// notFoundOr maps "no such row" Postgres outcomes to ErrNotFound and returns
// any other error unchanged.
func notFoundOr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == pgInvalidTextRepresentation {
		return ErrNotFound
	}
	return err
}

// requireRowsAffected turns an UPDATE that matched nothing into ErrNotFound.
func requireRowsAffected(tag pgconn.CommandTag) error {
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}
