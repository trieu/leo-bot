from __future__ import annotations

from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

@task()
def hello(name: str = "World") -> None:
    """Prints a greeting using the provided name parameter."""
    print("==========================================")
    print(f"Hello from Airflow!")
    print(f"The submitted 'name' parameter is: {name}")
    print(f"Today is {datetime.now()}")
    print("==========================================")


with DAG(
    dag_id='hello_world_taskflow_with_param',
    default_args=default_args,
    description='A TaskFlow DAG to print a parameter.',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=['taskflow', 'param'],
    params={
        "name": Param(
            default="World",
            type="string",
            title="Greeting Name",
            description="The name for the greeting."
        )
    },
) as dag:

    # ✅ Dynamically pull DAG param at runtime
    hello(name="{{ params.name }}")
