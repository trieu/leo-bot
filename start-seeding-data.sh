#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ENV_FILE="$PROJECT_ROOT/.env"
SEED_FILE="$PROJECT_ROOT/sql_scripts/sample_places.sql"
PG_CONTAINER_NAME="${PG_CONTAINER_NAME:-pgsql18_vector}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "❌ Environment file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$SEED_FILE" ]]; then
  echo "❌ Seed SQL file not found: $SEED_FILE" >&2
  exit 1
fi

DATABASE_URL="${PG_DSN:-${POSTGRES_URL:-}}"
if [[ -z "$DATABASE_URL" ]]; then
  echo "❌ PG_DSN or POSTGRES_URL must be configured in $ENV_FILE" >&2
  exit 1
fi

echo "🌱 Seeding places from $SEED_FILE..."

if command -v psql >/dev/null 2>&1; then
  psql "$DATABASE_URL" \
    -v ON_ERROR_STOP=1 \
    -f "$SEED_FILE"
elif command -v docker >/dev/null 2>&1 \
    && docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER_NAME"; then
  docker exec -i -u postgres "$PG_CONTAINER_NAME" \
    psql -v ON_ERROR_STOP=1 -d "${PG_DATABASE:-leo360}" \
    < "$SEED_FILE"
else
  echo "❌ Neither a usable psql client nor a running PostgreSQL Docker container was found." >&2
  exit 1
fi

echo "✅ Seed data applied successfully."