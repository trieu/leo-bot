#!/bin/bash

# --- Config ---
CONTAINER_NAME="pgsql16_vector"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
DEFAULT_DB="postgres"
TARGET_DB="customer360"
HOST_PORT=5432
DATA_VOLUME="pgdata_vector"

# --- Check if container exists ---
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  # If exists but not running, start it
  if ! docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "🔄 Starting existing container '${CONTAINER_NAME}'..."
    docker start "$CONTAINER_NAME"
    sleep 4
  else
    echo "🟢 PostgreSQL container '${CONTAINER_NAME}' is already running."
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

  echo "⏳ Waiting for PostgreSQL to start..."
  sleep 5

  # Install pgvector inside the container
  echo "📦 Installing pgvector extension..."
  docker exec -u root $CONTAINER_NAME bash -c "apt-get update && apt-get install -y postgresql-16-pgvector"
fi

# --- Create DB if not exists ---
docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -tc "SELECT 1 FROM pg_database WHERE datname = '${TARGET_DB}'" | grep -q 1 || \
  docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -c "CREATE DATABASE ${TARGET_DB};"

# --- Enable extensions ---
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# --- Create chat_messages table ---
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    persona_id VARCHAR(36),
    touchpoint_id VARCHAR(36),
    role TEXT CHECK (role IN ('user', 'bot')),
    message TEXT NOT NULL,
    message_hash TEXT NOT NULL,
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE (user_id, message_hash)
);
"

# --- Create chat_history_embeddings table ---
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "
CREATE TABLE IF NOT EXISTS chat_history_embeddings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    persona_id VARCHAR(36),
    touchpoint_id VARCHAR(36),
    role TEXT CHECK (role IN ('user', 'bot')),
    message TEXT,
    keywords TEXT[],
    embedding vector(768),
    created_at TIMESTAMP DEFAULT now()
);
"

# --- Create vector index if not exists ---
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "
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
"

# --- Create places table (with PostGIS + PlusCode) ---
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "
CREATE TABLE IF NOT EXISTS places (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
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
"

echo "✅ PostgreSQL 16 + PostGIS + pgvector is ready."
echo "   ➜ DB: customer360"
echo "   ➜ Tables: chat_messages, chat_history_embeddings, places"
echo "   ➜ Extensions: vector, postgis"
echo "   ➜ Port: $HOST_PORT"
