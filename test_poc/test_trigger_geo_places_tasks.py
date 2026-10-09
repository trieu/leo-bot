"""Trigger the church_places pipeline assets on a running Dagster webserver.

Start the cluster first (``./start_dagster.sh --cluster``), then run:

    env/bin/python test_poc/test_trigger_geo_places_tasks.py

Use ``--dry-run`` to validate the run config locally without contacting the
webserver. A real run calls Brave Place Search, so it may consume API quota.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from dagster import DagsterRunStatus, validate_run_config
from dagster_graphql import DagsterGraphQLClient

from dags_pipelines import defs

JOB_NAME = "geo_places_pipeline"
ASSET_NAMES = [
    "church_places",
    "church_brave_search",
    "church_mass_schedule",
    "church_knowledge",
]
CHURCH_PLACES_CONFIG: dict[str, Any] = {
    "name": "church",
    "latitude": 10.7536097,
    "longitude": 106.6284595,
    "radius": 6000,
    "count": 5,
}
# The other assets define defaults for every field, so only church_places needs config.
RUN_CONFIG: dict[str, Any] = {"ops": {"church_places": {"config": CHURCH_PLACES_CONFIG}}}
TERMINAL_STATUSES = {
    DagsterRunStatus.SUCCESS,
    DagsterRunStatus.FAILURE,
    DagsterRunStatus.CANCELED,
}


def validate_church_places_config() -> None:
    """Check the run config against the job definition without any network call."""
    validate_run_config(defs.resolve_job_def(JOB_NAME), RUN_CONFIG)


def trigger_church_places(
    host: str,
    port: int,
    repository_location: str,
    repository: str,
    timeout_seconds: int,
    poll_seconds: int = 5,
) -> DagsterRunStatus:
    """Launch a run that materializes all church assets and wait for it to finish."""
    client = DagsterGraphQLClient(host, port_number=port)
    run_id = client.submit_job_execution(
        JOB_NAME,
        repository_location_name=repository_location,
        repository_name=repository,
        run_config=RUN_CONFIG,
        asset_selection=ASSET_NAMES,
    )
    print(f"Submitted {JOB_NAME} run {run_id} for {', '.join(ASSET_NAMES)}.")

    deadline = time.monotonic() + timeout_seconds
    while True:
        status = client.get_run_status(run_id)
        if status in TERMINAL_STATUSES:
            print(f"Run {run_id} finished with status {status.value}.")
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Run {run_id} still {status.value} after {timeout_seconds}s; "
                "check the Dagster UI for its progress."
            )
        time.sleep(poll_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger all church assets on Dagster.")
    parser.add_argument("--dry-run", action="store_true", help="validate config only")
    parser.add_argument(
        "--host", default=os.getenv("DAGSTER_HOST", "localhost"),
        help="Dagster webserver host (default: DAGSTER_HOST or localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("DAGSTER_WEB_PORT", "3000")),
        help="Dagster webserver port (default: DAGSTER_WEB_PORT or 3000)",
    )
    parser.add_argument(
        "--repository-location", default="dags_pipelines",
        help="code location name, the module passed to dagster -m",
    )
    parser.add_argument(
        "--repository", default="__repository__",
        help="repository name inside the code location",
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds to wait for the run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_church_places_config()
    print(f"Run config is valid for {JOB_NAME}; selecting {', '.join(ASSET_NAMES)}.")
    if args.dry_run:
        return 0

    status = trigger_church_places(
        host=args.host,
        port=args.port,
        repository_location=args.repository_location,
        repository=args.repository,
        timeout_seconds=args.timeout,
    )
    return 0 if status == DagsterRunStatus.SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
