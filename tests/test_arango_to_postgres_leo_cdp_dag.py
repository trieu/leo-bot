import unittest
from airflow.models.dagbag import DagBag

# --- Configuration ---
DAG_ID = 'arango_to_postgres_leo_cdp'
# Adjust this path to point to your dags folder from where you run the test
DAG_FOLDER_PATH = 'dags/'
EXPECTED_TASK_COUNT = 8
EXPECTED_TASK_IDS = [
    'extract_profiles',
    'extract_transactions',
    'transform_and_embed_profiles',
    'transform_and_embed_txns',
    'load_profiles_to_pg',
    'load_txns_to_pg',
    'collect_tenants_from_profiles',
    'refresh_metrics_task'
]

class TestLeoCdpDag(unittest.TestCase):
    """
    Unit tests for the 'arango_to_postgres_leo_cdp' DAG.
    It checks for:
    1. DAG loading errors.
    2. Correct number of tasks.
    3. Correct task IDs.
    4. Correct task dependencies (the execution graph).
    """

    @classmethod
    def setUpClass(cls):
        """Load the DAG bag only once for all tests."""
        print(f"Loading DAGs from: {DAG_FOLDER_PATH}")
        cls.dagbag = DagBag(dag_folder=DAG_FOLDER_PATH, include_examples=False)
        cls.dag = cls.dagbag.get_dag(dag_id=DAG_ID)

    def test_dag_loading(self):
        """
        Test that the DAG file can be parsed without errors and the DAG exists.
        """
        # 1. Check for DAG import errors
        self.assertEqual(
            len(self.dagbag.import_errors), 0,
            f"DAG import errors found: {self.dagbag.import_errors}"
        )
        
        # 2. Check that the DAG was actually loaded
        self.assertIsNotNone(self.dag, f"DAG '{DAG_ID}' not found in DagBag.")
        
        # 3. Check that it's the correct DAG
        self.assertEqual(self.dag.dag_id, DAG_ID)

    def test_task_count_and_ids(self):
        """
        Test that the DAG has the expected number of tasks and correct task IDs.
        """
        self.assertIsNotNone(self.dag, "DAG not loaded")
        self.assertEqual(len(self.dag.tasks), EXPECTED_TASK_COUNT)
        
        actual_task_ids = [task.task_id for task in self.dag.tasks]
        # assertCountEqual checks for same elements, regardless of order
        self.assertCountEqual(actual_task_ids, EXPECTED_TASK_IDS)

    def _assert_task_dependencies(self, task_id, expected_upstream, expected_downstream):
        """Helper function to test upstream and downstream for a single task."""
        self.assertIsNotNone(self.dag, "DAG not loaded")
        task = self.dag.get_task(task_id)
        self.assertIsNotNone(task, f"Task '{task_id}' not found in DAG.")
        
        upstream_ids = [t.task_id for t in task.upstream_list]
        downstream_ids = [t.task_id for t in task.downstream_list]
        
        self.assertCountEqual(
            upstream_ids, expected_upstream,
            f"Incorrect upstream tasks for '{task_id}'"
        )
        self.assertCountEqual(
            downstream_ids, expected_downstream,
            f"Incorrect downstream tasks for '{task_id}'"
        )

    def test_all_dependencies(self):
        """
        Tests the complete dependency graph for the DAG.
        """
        # Start of the chain
        self._assert_task_dependencies(
            'extract_profiles',
            expected_upstream=[],
            expected_downstream=['extract_transactions', 'transform_and_embed_profiles']
        )
        
        # Middle of the chain (profiles)
        self._assert_task_dependencies(
            'transform_and_embed_profiles',
            expected_upstream=['extract_profiles'],
            expected_downstream=['load_profiles_to_pg', 'collect_tenants_from_profiles']
        )
        
        # Middle of the chain (transactions)
        self._assert_task_dependencies(
            'extract_transactions',
            expected_upstream=['extract_profiles'],
            expected_downstream=['transform_and_embed_txns']
        )
        
        self._assert_task_dependencies(
            'transform_and_embed_txns',
            expected_upstream=['extract_transactions'],
            expected_downstream=['load_txns_to_pg']
        )
        
        # Load tasks (parallel)
        self._assert_task_dependencies(
            'load_profiles_to_pg',
            expected_upstream=['transform_and_embed_profiles'],
            expected_downstream=['refresh_metrics_task']
        )
        
        self._assert_task_dependencies(
            'load_txns_to_pg',
            expected_upstream=['transform_and_embed_txns'],
            expected_downstream=['refresh_metrics_task']
        )
        
        # Tenant collection for refresh
        self._assert_task_dependencies(
            'collect_tenants_from_profiles',
            expected_upstream=['transform_and_embed_profiles'],
            expected_downstream=['refresh_metrics_task']
        )
        
        # Final refresh task (depends on both loads and the tenant list)
        self._assert_task_dependencies(
            'refresh_metrics_task',
            expected_upstream=['load_profiles_to_pg', 'load_txns_to_pg', 'collect_tenants_from_profiles'],
            expected_downstream=[]
        )


if __name__ == '__main__':
    unittest.main()