from dagster import Definitions, validate_run_config

from dags_pipelines import defs
from dags_pipelines.geo_places_pipeline import weekly


def test_pipeline_definitions_load():
    Definitions.validate_loadable(defs)


def test_pipeline_jobs_are_registered():
    assert {
        defs.resolve_job_def(name).name
        for name in (
        "leo_cdp_to_leo_bot_etl",
        "hello_world",
        "geo_places_pipeline",
        )
    }


def test_geo_places_pipeline_persists_to_postgres_without_csv_export():
    assert set(defs.resolve_job_def("geo_places_pipeline").graph.node_names()) == {
        "process_places",
        "process_brave_search",
        "process_mass_schedule",
        "process_knowledge",
    }


def test_geo_places_pipeline_runs_weekly():
    assert weekly.name == "geo_places_full_refresh_weekly"
    assert weekly.cron_schedule == "0 2 * * 0"


def test_arango_pipeline_contains_expected_ops():
    etl_job = defs.resolve_job_def("leo_cdp_to_leo_bot_etl")
    assert set(etl_job.graph.node_names()) == {
        "extract_profiles",
        "extract_transactions",
        "transform_and_embed_profiles",
        "transform_and_embed_txns",
        "load_profiles_to_pg",
        "load_txns_to_pg",
        "collect_tenants_from_profiles",
        "refresh_metrics_task",
    }
    dependencies = {
        node.name: {dependency.node for dependency in inputs.values()}
        for node, inputs in etl_job.graph.dependencies.items()
    }
    assert dependencies["extract_profiles"] == set()
    assert dependencies["extract_transactions"] == {"extract_profiles"}
    assert dependencies["transform_and_embed_profiles"] == {"extract_profiles"}
    assert dependencies["transform_and_embed_txns"] == {"extract_transactions"}
    assert dependencies["load_profiles_to_pg"] == {"transform_and_embed_profiles"}
    assert dependencies["load_txns_to_pg"] == {"transform_and_embed_txns"}
    assert dependencies["collect_tenants_from_profiles"] == {
        "transform_and_embed_profiles"
    }
    assert dependencies["refresh_metrics_task"] == {
        "collect_tenants_from_profiles",
        "load_profiles_to_pg",
        "load_txns_to_pg",
    }


def test_arango_pipeline_run_config_validates(monkeypatch):
    for variable in (
        "ARANGO_URL",
        "ARANGO_USERNAME",
        "ARANGO_PASSWORD",
        "PGSQL_DB_URL",
    ):
        monkeypatch.setenv(variable, "validation-placeholder")
    for variable in (
        "ARANGO_DATABASE",
        "ARANGO_PROFILE_COLLECTION",
        "ARANGO_TRANSACTION_COLLECTION",
    ):
        monkeypatch.delenv(variable, raising=False)

    validate_run_config(
        defs.resolve_job_def("leo_cdp_to_leo_bot_etl"),
        {
            "ops": {
                "extract_profiles": {
                    "config": {"segment_id": "test-segment", "batch_size": 5000}
                },
                "transform_and_embed_profiles": {"config": {"batch_size": 64}},
                "transform_and_embed_txns": {"config": {"batch_size": 64}},
            }
        },
    )


def test_hello_world_job_accepts_name_config():
    result = defs.resolve_job_def("hello_world").execute_in_process(
        run_config={"ops": {"hello": {"config": {"name": "Dagster"}}}}
    )
    assert result.success
