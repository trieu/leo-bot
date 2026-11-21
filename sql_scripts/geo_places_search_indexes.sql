-- Run outside a transaction on an existing deployment to avoid blocking writers.
CREATE INDEX CONCURRENTLY IF NOT EXISTS places_name_search_gin
ON geo_places USING GIN (to_tsvector('simple', coalesce(name, '')));

CREATE INDEX CONCURRENTLY IF NOT EXISTS places_text_search_gin
ON geo_places USING GIN (
    to_tsvector(
        'simple',
        coalesce(name, '') || ' ' || coalesce(description, '') || ' ' ||
        coalesce(category, '')
    )
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS knowledge_sources_geo_place_lookup_ix
ON knowledge_sources (tenant_id, (metadata->>'geo_place_id'))
WHERE status = 'active'
  AND (metadata->>'source' = 'geo_places'
       OR metadata->>'provider' = 'brave_search');

-- Apply the nullable coordinate contract to databases created from an older
-- version of this schema as well as to fresh installations above.
DO $$
BEGIN
    ALTER TABLE touchpoints
        ALTER COLUMN latitude DROP NOT NULL,
        ALTER COLUMN longitude DROP NOT NULL,
        ALTER COLUMN geom DROP NOT NULL;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'touchpoints'
          AND column_name = 'full_url'
    ) THEN
        ALTER TABLE touchpoints ADD COLUMN full_url TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'touchpoints'::regclass
          AND conname = 'touchpoints_coordinates_together_chk'
    ) THEN
        ALTER TABLE touchpoints
            ADD CONSTRAINT touchpoints_coordinates_together_chk CHECK (
                (latitude IS NULL AND longitude IS NULL AND geom IS NULL)
                OR (latitude IS NOT NULL AND longitude IS NOT NULL AND geom IS NOT NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'touchpoints'::regclass
          AND conname = 'touchpoints_web_qr_nullable_chk'
    ) THEN
        ALTER TABLE touchpoints
            ADD CONSTRAINT touchpoints_web_qr_nullable_chk CHECK (
                latitude IS NOT NULL
                OR type IS NOT DISTINCT FROM 'web_qr'
            );
    END IF;
END
$$;