#!/bin/bash

# --- Config ---
CONTAINER_NAME="pgsql16_vector"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
DEFAULT_DB="postgres"
TARGET_DB="customer360"
HOST_PORT=5432
DATA_VOLUME="pgdata_vector"

# --- Check if container is running ---
if docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  echo "🟢 PostgreSQL container '${CONTAINER_NAME}' is already running."
else
  # Create volume if not exists
  if ! docker volume ls | grep -q "$DATA_VOLUME"; then
    docker volume create "$DATA_VOLUME"
  fi

  # Start pgvector image with PostgreSQL 16 + pgvector 0.8.0
  docker run -d \
    --name $CONTAINER_NAME \
    -e POSTGRES_USER=$POSTGRES_USER \
    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
    -e POSTGRES_DB=$DEFAULT_DB \
    -p $HOST_PORT:5432 \
    -v $DATA_VOLUME:/var/lib/postgresql/data \
    pgvector/pgvector:0.8.0-pg16

  echo "⏳ Waiting for PostgreSQL to start..."
  sleep 5
fi

# --- Create extension + database + table ---
docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -tc "SELECT 1 FROM pg_database WHERE datname = '${TARGET_DB}'" | grep -q 1 || \
  docker exec -u postgres $CONTAINER_NAME psql -d $DEFAULT_DB -c "CREATE DATABASE ${TARGET_DB};"

# Enable vector extension
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create chat_history_embeddings table
docker exec -u postgres $CONTAINER_NAME psql -d $TARGET_DB -c "
CREATE TABLE IF NOT EXISTS chat_history_embeddings (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    role TEXT CHECK (role IN ('user', 'bot')),
    message TEXT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT now()
);
"

# Create index if not exists (guarded by plpgsql DO block)
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

echo "✅ PostgreSQL 16 with pgvector is ready."
echo "   ➜ DB: customer360"
echo "   ➜ Table: chat_history_embeddings"
echo "   ➜ Port: $HOST_PORT"