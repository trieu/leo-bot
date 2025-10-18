from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# These arguments will be passed to your DAG
# You can override them on a task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# The DAG context manager automatically creates the DAG object.
# The 'schedule' argument is set to None to run it manually.
with DAG(
    dag_id='hello_world_test_dag',
    default_args=default_args,
    description='A simple DAG to test Airflow installation',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=['test', 'example'],
) as dag:
    
    # Task 1: Print "Hello World" using the BashOperator
    # This task executes the given bash command.
    hello_task = BashOperator(
        task_id='print_hello_message',
        bash_command='echo "==========================================" && echo "Hello from Airflow! The system is operational." && echo "Today is $(date)" && echo "=========================================="',
    )
    
    # Since we only have one task, the DAG definition is complete.
    # If you had more tasks, you would define their dependencies here, like:
    # task_a >> task_b >> task_c
