#!/bin/bash

# --- Config ---
CONTAINER_NAME="pgsql16_vector"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
DEFAULT_DB="postgres"
TARGET_DB="customer360"
HOST_PORT=5432
DATA_VOLUME="pgdata_vector"
SCHEMA_VERSION=1  # Version 0.1 (Migration 1)

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
  local sql=$3
  echo "🚀 Applying migration for version $version: $description..."
  docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "$sql" || {
    echo "❌ Error: Failed to apply migration version $version."
    exit 1
  }
  docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "INSERT INTO schema_migrations (version, description) VALUES ($version, '$description');" || {
    echo "❌ Error: Failed to record migration version $version."
    exit 1
  }
}

# --- Migration 1: Initial schema with chat tables, places (with hash-based id), and system_users ---
if [ "$CURRENT_VERSION" -lt 1 ]; then
  apply_migration 1 "Initial schema with chat tables, places (hash-based id), and system_users" "
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL,
        tenant_id TEXT NOT NULL,
        persona_id VARCHAR(36),
        touchpoint_id VARCHAR(36),
        role TEXT CHECK (role IN ('user', 'bot')),
        message TEXT NOT NULL,
        message_hash TEXT NOT NULL,
        keywords TEXT[],
        created_at TIMESTAMP DEFAULT now(),
        UNIQUE (user_id, message_hash)
    );

    CREATE TABLE IF NOT EXISTS chat_history_embeddings (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL,
        tenant_id TEXT NOT NULL,
        persona_id VARCHAR(36),
        touchpoint_id VARCHAR(36),
        role TEXT CHECK (role IN ('user', 'bot')),
        message TEXT,
        keywords TEXT[],
        embedding vector(768),
        created_at TIMESTAMP DEFAULT now()
    );

    DO \$\$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'chat_history_embeddings'
          AND indexname = 'chat_history_embeddings_embedding_idx'
      ) THEN
        EXECUTE 'CREATE INDEX chat_history_embeddings_embedding_idx
                 ON chat_history_embeddings
                 USING ivfflat (embedding vector_cosine_ops)
                 WITH (lists = 100);';
      END IF;
    END
    \$\$;

    CREATE TABLE IF NOT EXISTS places (
        id BIGINT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        address TEXT,
        description TEXT,
        category TEXT,
        tags TEXT[],
        pluscode TEXT UNIQUE,
        geom GEOMETRY(Point, 4326) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_places_geom ON places USING GIST (geom);
    CREATE INDEX IF NOT EXISTS idx_places_pluscode ON places (pluscode);

    CREATE TABLE IF NOT EXISTS system_users (
        id SERIAL PRIMARY KEY,
        activation_key VARCHAR(64),
        avatar_url TEXT,
        creation_time BIGINT NOT NULL,
        custom_data JSONB,
        display_name TEXT NOT NULL,
        is_online BOOLEAN DEFAULT FALSE,
        modification_time BIGINT,
        tenant_id TEXT NOT NULL,
        registered_time BIGINT DEFAULT 0,
        role INTEGER NOT NULL,
        status INTEGER NOT NULL,
        user_email TEXT UNIQUE NOT NULL,
        user_login TEXT UNIQUE NOT NULL,
        user_pass TEXT NOT NULL,
        access_profile_fields TEXT[],
        action_logs TEXT[],
        in_groups TEXT[],
        business_unit TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT check_display_name_not_empty CHECK (display_name <> ''),
        CONSTRAINT check_user_email_not_empty CHECK (user_email <> ''),
        CONSTRAINT check_user_login_not_empty CHECK (user_login <> ''),
        CONSTRAINT check_user_pass_not_empty CHECK (user_pass <> '')
    );
    CREATE INDEX IF NOT EXISTS idx_system_users_user_email ON system_users (user_email);
    CREATE INDEX IF NOT EXISTS idx_system_users_user_login ON system_users (user_login);
    CREATE INDEX IF NOT EXISTS idx_system_users_tenant_id ON system_users (tenant_id);
  "
fi

# --- Verify all tables exist ---
TABLES=("chat_messages" "chat_history_embeddings" "places" "schema_migrations" "system_users")
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