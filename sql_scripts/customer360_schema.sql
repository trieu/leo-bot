---------------- the SQL DDL Schema for LEO BOT -----------------
-----------------------------------------------------------------
-- ============================================================
-- Enable required extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- ENUM TYPES
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chat_status') THEN
        CREATE TYPE chat_status AS ENUM ('active', 'closed', 'escalated', 'archived');
    END IF;
END$$;

-- ============================================================
-- Chat Messages
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    message_hash TEXT PRIMARY KEY,                 
    user_id VARCHAR(36) NOT NULL,
    cdp_profile_id VARCHAR(36),
    tenant_id VARCHAR(36) NOT NULL,
    persona_id VARCHAR(36),
    touchpoint_id VARCHAR(36),
    channel VARCHAR(50) NOT NULL DEFAULT 'webchat',
    status chat_status DEFAULT 'active',
    role TEXT CHECK (role IN ('user', 'bot')),
    message TEXT NOT NULL,
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_intent_label VARCHAR(255),
    last_intent_confidence NUMERIC(5, 4) CHECK (last_intent_confidence >= 0 AND last_intent_confidence <= 1),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, message_hash)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created_at
    ON chat_messages (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_cdp_profile
    ON chat_messages (cdp_profile_id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_tenant_role
    ON chat_messages (tenant_id, role);

CREATE INDEX IF NOT EXISTS idx_chat_messages_tsv
    ON chat_messages USING GIN (to_tsvector('english', message));

-- ============================================================
-- Chat Message Embeddings (Multi-tenant Aware)
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_message_embeddings (
    message_hash TEXT PRIMARY KEY REFERENCES chat_messages(message_hash) ON DELETE CASCADE,
    tenant_id VARCHAR(36) NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector similarity
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'chat_message_embeddings'
          AND indexname = 'chat_message_embeddings_embedding_idx'
    ) THEN
        EXECUTE '
            CREATE INDEX chat_message_embeddings_embedding_idx
            ON chat_message_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 200);
        ';
    END IF;
END $$;


-- Defines the type of knowledge source (e.g., book, report, dataset, etc.)
CREATE TYPE knowledge_source_type AS ENUM (
    -- Textual & Document Sources
    'book_summary',               -- extracted or summarized from books
    'report_analytics',           -- business or market reports
    'uploaded_document',          -- user-uploaded PDFs, Word docs, etc.
    'web_page',                   -- scraped website content
    'research_paper',             -- scientific or academic publication
    'knowledge_base_article',     -- internal or external wiki, FAQ, SOP

    -- Data & Technical Sources
    'dataset',                    -- structured tabular data (CSV, JSON, SQL)
    'code_repository',            -- source code or API docs
    'api_documentation',          -- REST/GraphQL API reference or schema
    'system_log',                 -- application or infrastructure logs

    -- Conversational & Social Sources
    'conversation_log',           -- chatbot or customer support transcripts
    'meeting_transcript',         -- AI-generated meeting notes or Zoom calls
    'social_media_post',          -- tweets, LinkedIn posts, or public threads

    -- Media & Multimodal Sources
    'video_transcript',           -- text extracted from video
    'audio_transcript',           -- text extracted from podcast or call
    'other'                       -- fallback for anything unclassified
);



-- Tracks the state of the document in the processing pipeline
CREATE TYPE processing_status AS ENUM (
    'pending',      -- Waiting to be processed
    'processing',   -- Actively being chunked and embedded
    'active',       -- Ready for querying
    'failed',       -- An error occurred during processing
    'archived'      -- No longer in active use
);

-- ============================================================
-- Knowledge Sources
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(36) NOT NULL,
    tenant_id VARCHAR(36) NOT NULL,
    source_type knowledge_source_type DEFAULT 'other',
    name TEXT NOT NULL, -- e.g., 'Q3 Financial Report.pdf' or 'The Great Gatsby Summary'
    code_name VARCHAR(50) DEFAULT '',
    uri TEXT,           -- Optional: Path to the original file in blob storage (e.g., s3://bucket/file.md)
    status processing_status NOT NULL DEFAULT 'pending',
    metadata JSONB,     -- Flexible field for extra info like author, source URL, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Knowledge Chunks
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    content TEXT NOT NULL,          -- The actual text chunk
    embedding VECTOR(768) NOT NULL, -- The vector embedding for the content
    chunk_sequence INT,             -- The order of this chunk within the original document
    metadata JSONB,                 -- Extra info like page number, section headers, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index to quickly retrieve all chunks for a given source document
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source
    ON knowledge_chunks (source_id);

-- This is the crucial index for fast similarity searches
-- Using IVFFlat to be consistent with your example. HNSW is another excellent option.
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100); -- The 'lists' parameter should be tuned based on your table size.

-- Optional: A GIN index can be useful for filtering by metadata before a vector search
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_metadata
    ON knowledge_chunks USING GIN (metadata jsonb_path_ops);

-- Index for quickly finding all sources for a specific user or tenant
CREATE INDEX IF NOT EXISTS idx_knowledge_sources_user_tenant
    ON knowledge_sources (user_id, tenant_id);

-- Index to efficiently query sources by their processing status
CREATE INDEX IF NOT EXISTS idx_knowledge_sources_status
    ON knowledge_sources (status);

-- ============================================================
-- Places (Geo-aware data)
-- ============================================================
CREATE TABLE IF NOT EXISTS places (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    address TEXT,
    description TEXT,
    category TEXT,
    tags TEXT[],
    pluscode TEXT UNIQUE,
    geom GEOMETRY(Point, 4326) NOT NULL,
    region_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_places_geom ON places USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_places_pluscode ON places (pluscode);
CREATE INDEX IF NOT EXISTS idx_places_region ON places (region_id);

-- ============================================================
-- System Users
-- ============================================================
CREATE TABLE IF NOT EXISTS system_users (
    id SERIAL PRIMARY KEY,
    activation_key VARCHAR(64),
    avatar_url TEXT,
    creation_time BIGINT NOT NULL,
    custom_data JSONB,
    display_name TEXT NOT NULL,
    is_online BOOLEAN DEFAULT FALSE,
    modification_time BIGINT,
    tenant_id VARCHAR(36) NOT NULL,
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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT check_display_name_not_empty CHECK (display_name <> ''),
    CONSTRAINT check_user_email_not_empty CHECK (user_email <> ''),
    CONSTRAINT check_user_login_not_empty CHECK (user_login <> ''),
    CONSTRAINT check_user_pass_not_empty CHECK (user_pass <> '')
);

CREATE INDEX IF NOT EXISTS idx_system_users_user_email ON system_users (user_email);
CREATE INDEX IF NOT EXISTS idx_system_users_user_login ON system_users (user_login);
CREATE INDEX IF NOT EXISTS idx_system_users_tenant_id ON system_users (tenant_id);
CREATE INDEX IF NOT EXISTS idx_system_users_custom_data ON system_users USING GIN (custom_data jsonb_path_ops);

-- ============================================================
-- Conversational Context
-- ============================================================
CREATE TABLE IF NOT EXISTS conversational_context (
    user_id VARCHAR(36) NOT NULL,
    touchpoint_id VARCHAR(36) NOT NULL,
    cdp_profile_id VARCHAR(36),
    context_data JSONB NOT NULL,
    embedding VECTOR(768),
    intent_label VARCHAR(255),
    intent_confidence NUMERIC(5, 4) CHECK (intent_confidence >= 0 AND intent_confidence <= 1) DEFAULT 0,
    updated_by TEXT DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, touchpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_conversational_context_jsonb
    ON conversational_context USING GIN (context_data jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_conversational_context_cdp_profile
    ON conversational_context (cdp_profile_id);

CREATE INDEX IF NOT EXISTS idx_conversational_context_user
    ON conversational_context (user_id);

CREATE INDEX IF NOT EXISTS idx_conversational_context_embedding
    ON conversational_context USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_conversational_context_intent
    ON conversational_context (intent_label);

-- ============================================================
-- Triggers for automatic updated_at maintenance
-- ============================================================
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing triggers if they exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_chat_messages_timestamp') THEN
        EXECUTE 'DROP TRIGGER trg_chat_messages_timestamp ON chat_messages';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_places_timestamp') THEN
        EXECUTE 'DROP TRIGGER trg_places_timestamp ON places';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_system_users_timestamp') THEN
        EXECUTE 'DROP TRIGGER trg_system_users_timestamp ON system_users';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_conversational_context_timestamp') THEN
        EXECUTE 'DROP TRIGGER trg_conversational_context_timestamp ON conversational_context';
    END IF;
END $$;

-- Recreate update triggers consistently
CREATE TRIGGER trg_chat_messages_timestamp
BEFORE UPDATE ON chat_messages
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trg_places_timestamp
BEFORE UPDATE ON places
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trg_system_users_timestamp
BEFORE UPDATE ON system_users
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trg_conversational_context_timestamp
BEFORE UPDATE ON conversational_context
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();
