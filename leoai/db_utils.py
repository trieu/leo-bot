from dotenv import load_dotenv
import os
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/customer360")


# --- DB Connection ---

def get_pg_conn():
    return psycopg.connect(DATABASE_URL)