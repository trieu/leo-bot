-- Enable extensions if needed
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

-- ==============================
-- Chat messages
-- ==============================
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
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (user_id, message_hash)
);

-- ==============================
-- Chat history embeddings
-- ==============================
CREATE TABLE IF NOT EXISTS chat_history_embeddings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    tenant_id TEXT NOT NULL,
    persona_id VARCHAR(36),
    touchpoint_id VARCHAR(36),
    role TEXT CHECK (role IN ('user', 'bot')),
    message TEXT,
    keywords TEXT[],
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'chat_history_embeddings'
          AND indexname = 'chat_history_embeddings_embedding_idx'
    ) THEN
        EXECUTE '
            CREATE INDEX chat_history_embeddings_embedding_idx
            ON chat_history_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        ';
    END IF;
END $$;

-- ==============================
-- Places
-- ==============================
CREATE TABLE IF NOT EXISTS places (
    id BIGSERIAL PRIMARY KEY,
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

-- ==============================
-- System users
-- ==============================
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

-- ==============================
-- Conversational context
-- ==============================
CREATE TABLE IF NOT EXISTS conversational_context (
    user_id VARCHAR(36) NOT NULL,
    touchpoint_id VARCHAR(36) NOT NULL,
    context_data JSONB NOT NULL,
    embedding VECTOR(768),
    intent_label VARCHAR(255),
    intent_confidence NUMERIC(5, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, touchpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_conversational_context_jsonb
    ON conversational_context USING GIN (context_data jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_conversational_context_user
    ON conversational_context (user_id);

CREATE INDEX IF NOT EXISTS idx_conversational_context_embedding
    ON conversational_context USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_conversational_context_intent
    ON conversational_context (intent_label);

-- Maintain updated_at automatically
CREATE OR REPLACE FUNCTION update_conversational_context_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER conversational_context_timestamp
BEFORE UPDATE ON conversational_context
FOR EACH ROW
EXECUTE FUNCTION update_conversational_context_timestamp();
