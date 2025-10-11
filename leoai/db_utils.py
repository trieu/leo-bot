import hashlib
from dotenv import load_dotenv
import os
import psycopg

load_dotenv(override=True)

DEFAULT_DATABASE_URL = '"postgresql://postgres:password@localhost:5432/customer360"'
DATABASE_URL = os.getenv("POSTGRES_URL", DEFAULT_DATABASE_URL )


# --- DB Connection ---

def get_pg_conn():
    """Synchronous connection"""
    return psycopg.connect(DATABASE_URL)


async def get_async_pg_conn():
    """Asynchronous connection"""
    return await psycopg.AsyncConnection.connect(DATABASE_URL)


# =====================================================================
# Utility functions
# =====================================================================

def sha256_hash(text: str) -> str:
    """Generate a SHA256 hash of text to detect duplicate messages."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
