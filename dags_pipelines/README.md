# Dagster pipelines

This package exposes one Dagster code location by merging the `defs` objects
from its three pipeline modules:

| Python module | Dagster definition | Purpose |
| --- | --- | --- |
| `arango_to_postgres_leo_cdp.py` | `leo_cdp_to_leo_bot_etl` job | Load segment profiles and linked transactions from ArangoDB into PostgreSQL |
| `geo_places_pipeline.py` | `geo_places_pipeline` asset job | Search and enrich geo places, then persist their knowledge |
| `hello_world_dag.py` | `hello_world` job | Small example of a config-driven Dagster op |

## ArangoDB to PostgreSQL ETL

The job runs these ops; profile and transaction processing branch after profile
extraction, then join at the metrics refresh:

```text
extract_profiles
├── extract_transactions ── transform_and_embed_txns ── load_txns_to_pg ──┐
└── transform_and_embed_profiles ── load_profiles_to_pg ──────────────────┤
                                └── collect_tenants_from_profiles ────────┴── refresh_metrics_task
```

`extract_profiles` selects active profiles whose `inSegments` contains the
requested `segment_id`. It paginates by `_key` in batches of 5,000. An empty
segment ID returns no profiles; it does not fail the run. Transactions are
selected from `cdp_profile2conversion` where `_from` points to one of those
profiles.

The transform ops build text summaries and request embeddings through
`leoai.ai_core.get_embedding_model()`. Each op submits texts in batches of 64
(configurable per op). Profile records preserve the source document in
`metadata`; transaction records preserve it in `context_data`.

The load ops upsert to:

- `customer_profile`, on `cdp_profile_id`
- `transactional_context`, on `(tenant_id, user_id, txn_id)`

After both loads finish, `refresh_metrics_task` calls
`refresh_customer_metrics(tenant_id)` for each tenant found in the profiles.
Every ETL op has one retry, delayed by five minutes. This job has no schedule;
launch it manually with a non-empty segment ID.

Example Dagster run config:

```yaml
ops:
  extract_profiles:
    config:
      segment_id: "your-segment-id"
      batch_size: 5000
  transform_and_embed_profiles:
    config:
      batch_size: 64
  transform_and_embed_txns:
    config:
      batch_size: 64
```

The embedding adapter uses `EMBEDDING_PROVIDER`, falling back to `AI_PROVIDER`
and then `google`. `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` override the
provider defaults; dimensions default to 768. The key is `EMBEDDING_API_KEY`
when set, otherwise it uses the key for the selected provider
(`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`).

## Geo places enrichment

The assets upsert to `geo_places` and create matching `knowledge_sources` and
`knowledge_chunks` records. Existing place UUIDs remain unchanged. Initialize
the development database from
[`../sql_scripts/leo360_schema.sql`](../sql_scripts/leo360_schema.sql) before
materializing these assets; the pipeline does not run schema migrations.
Asset dependencies are:

```text
church_places
├── church_brave_search ───┐
└── church_mass_schedule ──┴── church_knowledge
```

The assets do the following:

1. **`church_places`** searches with `BravePlaceSearchClient` and upserts by
   `geo_place_id`. The database primary key is `geo_places.id`, used as the
   internal `geo_place_id` when related knowledge records are stored. The search
   name, center latitude/longitude, and radius are
   required inputs supplied through the Dagster run config. Brave's radius is
   a location bias, not a strict distance cutoff.
   Every `geo_place_id` uses a source prefix such as `brave_api:`, `sample:`,
   `google_api:`, or `leo_crawler:`; the database rejects unprefixed IDs.
   A discovery result without coordinates is still stored with a NULL `geom`;
   `church_brave_search` performs a follow-up place search to recover
   coordinates before continuing enrichment.
2. **`church_brave_search`** searches Brave's grounded web context for one
   place at a time. Each place is committed to PostgreSQL before the next place
   is searched, so a later failure does not roll back earlier results. Each
   `grounding.generic` result is stored as a `knowledge_sources` row, and each
   result snippet is embedded and stored as a `knowledge_chunks` row only when
   the result contains the place-name tokens and church/parish/mass context.
   Unrelated tourism results are skipped. Places are revisited after 90 days by
   default.
3. **`church_mass_schedule`** fetches parish pages using the shared public-web
   fetcher, which validates public IPs and redirects. It follows only safe
   same-origin subpages and asks `AIClient` to extract schedules with page-text
   evidence. It processes and commits one place at a time.
4. **`church_knowledge`** uses `AIClient` to create a factual place description
   and tags, then writes the enriched place and its vectorized knowledge directly
   to PostgreSQL. It upserts one `knowledge_sources` record and its
   768-dimensional `knowledge_chunks` per place. It processes and commits one
   place at a time. Pipeline-owned sources use user `geo_places_pipeline` and
   tenant `default`.

The required search config and optional result count look like:

```yaml
ops:
  church_places:
    config:
      name: "church"
      latitude: 10.7536097
      longitude: 106.6284595
      radius: 6000
      count: 3
```

Other config defaults:

| Config | Defaults |
| --- | --- |
| `BraveSearchConfig` | Vietnamese search; 90-day refresh; 10 results per place; sequential processing |
| `MassScheduleConfig` | 60-day refresh; sequential processing; 0.6 minimum confidence; up to 3 subpages |
| `KnowledgeEnrichmentConfig` | Sequential processing; 200-token chunks; 40-token overlap |

The `church_full_refresh_weekly` schedule runs at `0 2 * * 0` (02:00 every
Sunday, in the Dagster deployment's timezone). Enable the
schedule in Dagster if it should run automatically.

## Greeting job

`hello_world` logs a greeting and the current time. Its `GreetingConfig.name`
defaults to `World`; override it with:

```yaml
ops:
  hello:
    config:
      name: "Dagster"
```

## Environment and database requirements

Set variables in the Dagster process environment. The pipeline does not
hard-code credentials: Dagster resources use `EnvVar` for their secrets, while
the shared embedding adapter reads its provider settings from the environment.

| Variable | Used by | Required/default |
| --- | --- | --- |
| `PGSQL_DB_URL` | Both database pipelines | Required PostgreSQL DSN |
| `ARANGO_URL` | ArangoDB ETL | Required ArangoDB endpoint |
| `ARANGO_USERNAME` | ArangoDB ETL | Required ArangoDB username |
| `ARANGO_PASSWORD` | ArangoDB ETL | Required ArangoDB password |
| `ARANGO_DATABASE` | ArangoDB ETL | Optional; `leo_cdp` |
| `ARANGO_PROFILE_COLLECTION` | ArangoDB ETL | Optional; `cdp_profile` |
| `ARANGO_TRANSACTION_COLLECTION` | ArangoDB ETL | Optional; `cdp_profile2conversion` |
| `BRAVE_API_KEY` | Church place discovery and grounded web search | Required |
| `AI_PROVIDER`, `AI_CHAT_MODEL` | `AIClient` description and schedule generation | Provider defaults apply |
| `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY` | `AIClient` generation and embedding | Required for the selected provider |
The embedding settings and provider credentials described above are also
required by the ETL's configured embedding provider. The geo places pipeline
requires PostgreSQL/PostGIS access and read/write permissions on `geo_places`,
`knowledge_sources`, and `knowledge_chunks`; its development assets do not
alter the schema.
The ArangoDB ETL expects `customer_profile`, `transactional_context`, and the
`refresh_customer_metrics(text)` function to exist.

## Run and test

From the repository root:

```bash
python -m pip install -r requirements.txt
./start_dagster.sh
```

`start_dagster.sh` starts `dockers/pgsql/start_pgsql_pgvector.sh` first. The
PostgreSQL helper creates `TARGET_DB` when it is missing; `TARGET_DB` defaults
to `leo360` and can be overridden through the environment. Pass
`--reset-db` to reset the PostgreSQL data before Dagster starts.

The script uses `env/bin/dagster` when available, otherwise `dagster` from
`PATH`. By default it starts the local Dagster development deployment. To run
the webserver and daemon as separate supervised processes, use cluster mode:

```bash
./start_dagster.sh --cluster
```

Cluster mode shares `DAGSTER_HOME` between both processes, defaults to
`0.0.0.0:3000`, writes separate webserver and daemon logs under `logs/`, and
commits each pipeline place independently. Override `DAGSTER_HOME`,
`DAGSTER_WEB_HOST`, `DAGSTER_WEB_PORT`, `DAGSTER_LOG_DIR`, or
`DAGSTER_PID_DIR` through the environment. Additional Dagster webserver options
can be passed after `--cluster`.

Use the Dagster UI at `http://localhost:3000` to launch the ETL or greeting job,
materialize geo place assets with the required search config, and enable the
weekly schedule. To check the code location and tests:

```bash
dagster definitions validate -m dags_pipelines
env/bin/python -m pytest -q tests/test_dagster_pipelines.py
```
