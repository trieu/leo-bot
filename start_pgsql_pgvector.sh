#!/bin/bash

# --- Config ---
CONTAINER_NAME="pgsql16_vector"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
DEFAULT_DB="postgres"
TARGET_DB="customer360"
HOST_PORT=5432
DATA_VOLUME="pgdata_vector"
SCHEMA_VERSION=4  # Version 0.1 (Migration 1)
SCHEMA_DESCRIPTION="init database; create conversational_context table"
SQL_FILE_PATH="./sql_scripts/customer360_schema.sql"

# --- Function to check PostgreSQL readiness ---
wait_for_postgres() {
  local max_attempts=10
  local attempt=1
  echo "⏳ Checking if PostgreSQL is ready..."
  until docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -c "SELECT 1;" >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
      echo "❌ Error: PostgreSQL is not ready after $max_attempts attempts."
      exit 1
    fi
    echo "⏳ Attempt $attempt/$max_attempts: Waiting for PostgreSQL to be ready..."
    sleep 2
    ((attempt++))
  done
  echo "🟢 PostgreSQL is ready."
}

# --- Check if container exists ---
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  # If exists but not running, start it
  if ! docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "🔄 Starting existing container '${CONTAINER_NAME}'..."
    docker start "$CONTAINER_NAME"
    wait_for_postgres
  else
    echo "🟢 PostgreSQL container '${CONTAINER_NAME}' is already running."
    wait_for_postgres
  fi
else
  # Create volume if needed
  if ! docker volume ls | grep -q "$DATA_VOLUME"; then
    docker volume create "$DATA_VOLUME"
  fi

  echo "🚀 Launching new PostgreSQL container '${CONTAINER_NAME}'..."
  docker run -d \
    --name $CONTAINER_NAME \
    -e POSTGRES_USER=$POSTGRES_USER \
    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
    -e POSTGRES_DB=$DEFAULT_DB \
    -p $HOST_PORT:5432 \
    -v $DATA_VOLUME:/var/lib/postgresql/data \
    postgis/postgis:16-3.5  # ✅ includes PostgreSQL 16 + PostGIS

  wait_for_postgres

  # Install pgvector inside the container
  echo "📦 Installing pgvector extension..."
  docker exec -u root $CONTAINER_NAME bash -c "apt-get update && apt-get install -y postgresql-16-pgvector"
fi

# --- Fix collation version mismatch ---
echo "🔧 Checking and fixing collation version mismatch for 'postgres' and 'template1'..."
docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -c "ALTER DATABASE postgres REFRESH COLLATION VERSION;" || {
  echo "⚠️ Warning: Failed to refresh collation version for 'postgres'. Continuing..."
}
docker exec -u postgres $CONTAINER_NAME psql -d template1 -c "ALTER DATABASE template1 REFRESH COLLATION VERSION;" || {
  echo "⚠️ Warning: Failed to refresh collation version for 'template1'. Continuing..."
}

# --- Create DB if not exists ---
echo "🔄 Checking if database '${TARGET_DB}' exists..."
docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -tc "SELECT 1 FROM pg_database WHERE datname = '${TARGET_DB}'" | grep -q 1 || {
  echo "🚀 Creating database '${TARGET_DB}'..."
  docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -c "CREATE DATABASE ${TARGET_DB};" || {
    echo "❌ Error: Failed to create database '${TARGET_DB}'."
    exit 1
  }
}

# --- Ensure connection to target database ---
wait_for_postgres_target() {
  local max_attempts=5
  local attempt=1
  echo "⏳ Checking if database '${TARGET_DB}' is accessible..."
  until docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "SELECT 1;" >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
      echo "❌ Error: Database '${TARGET_DB}' is not accessible after $max_attempts attempts."
      exit 1
    fi
    echo "⏳ Attempt $attempt/$max_attempts: Waiting for database '${TARGET_DB}' to be ready..."
    sleep 2
    ((attempt++))
  done
  echo "🟢 Database '${TARGET_DB}' is accessible."
}
wait_for_postgres_target

# --- Enable extensions ---
echo "🔧 Enabling extensions in '${TARGET_DB}'..."
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
  echo "❌ Error: Failed to enable 'vector' extension."
  exit 1
}
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "CREATE EXTENSION IF NOT EXISTS postgis;" || {
  echo "❌ Error: Failed to enable 'postgis' extension."
  exit 1
}

# --- Create schema_migrations table to track versions ---
echo "🔧 Creating schema_migrations table..."
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);
"

# --- Check current schema version ---
echo "🔍 Checking current schema version..."
CURRENT_VERSION=$(docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -t -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;" 2>/dev/null | tr -d '[:space:]' || echo "0")
if [ -z "$CURRENT_VERSION" ]; then
  CURRENT_VERSION=0
fi
echo "ℹ️ Current schema version: $CURRENT_VERSION"

# --- Function to apply migration ---
apply_migration() {
  local version=$1
  local description=$2
  local sql_file_path=$3

  echo "🚀 Applying migration for version $version: $description"

  # Check if the file exists before proceeding
  if [[ ! -f "$sql_file_path" ]]; then
    echo "❌ Error: SQL file not found at path: $sql_file_path"
    exit 1
  fi

  # Apply SQL migration from file
  docker exec -i -u postgres "$CONTAINER_NAME" psql -d "$TARGET_DB" < "$sql_file_path" || {
    echo "❌ Error: Failed to apply migration version $version from $sql_file_path."
    exit 1
  }

  # Record migration version in schema_migrations table
  docker exec -u postgres "$CONTAINER_NAME" psql -d "$TARGET_DB" -c \
    "INSERT INTO schema_migrations (version, description, applied_at) VALUES ($version, '$description', NOW());" || {
      echo "❌ Error: Failed to record migration version $version."
      exit 1
    }

  echo "✅ Migration $version applied successfully."
}


# --- Migration 1: Initial schema with chat tables, places (with hash-based id), and system_users ---
if [ $CURRENT_VERSION -lt $SCHEMA_VERSION ]; then
  apply_migration $SCHEMA_VERSION "$SCHEMA_DESCRIPTION" "$SQL_FILE_PATH"
fi

# --- Verify all tables exist ---
TABLES=("chat_messages" "chat_history_embeddings" "places" "schema_migrations" "system_users" "conversational_context")
for table in "${TABLES[@]}"; do
  docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -tc "SELECT 1 FROM pg_tables WHERE tablename = '$table'" | grep -q 1 || {
    echo "❌ Error: Table '$table' is missing after migrations."
    exit 1
  }
done

echo "✅ PostgreSQL 16 + PostGIS + pgvector is ready."
echo "   ➜ DB: customer360"
echo "   ➜ Tables: ${TABLES[*]}"
echo "   ➜ Extensions: vector, postgis"
echo "   ➜ Schema Version: $SCHEMA_VERSION"
echo "   ➜ Port: $HOST_PORT"