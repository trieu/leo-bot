import hashlib
from dotenv import load_dotenv
import os
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/customer360")


# --- DB Connection ---

def get_pg_conn():
    return psycopg.connect(DATABASE_URL)


# =====================================================================
# Utility functions
# =====================================================================

def sha256_hash(text: str) -> str:
    """Generate a SHA256 hash of text to detect duplicate messages."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()