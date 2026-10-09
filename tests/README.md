
# Dagster pipeline definitions and dependency graph:
env/bin/python -m pytest -q tests/test_dagster_pipelines.py

## Recommendation PoC

Run offline regression tests (no hosted AI or database requests):

```bash
env/bin/python -m pytest -q tests/test_recommendation.py
```

To additionally validate PostgreSQL/pgvector using `PGSQL_DB_URL` (or `PGSQL_DB_URL`):

```bash
RUN_RECOMMENDATION_DB_TESTS=1 env/bin/python -m pytest -q tests/test_recommendation.py
```

The database test uses fake embeddings and a unique schema inside a transaction,
then rolls it back; it does not change existing product/profile tables.