import pytest
from main_config import setup_logging

setup_logging()

def test_run_profile_queue():
    assert pytest.main([
        "--log-cli-level=INFO",
        "-s",
        "tests/test_leocdp_save_profile.py"
    ]) == 0
