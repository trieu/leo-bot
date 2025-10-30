# async_pgvector_recommend.py
"""
Async pgvector migration & recommender.

Requirements:
  pip install asyncpg sentence-transformers numpy

Make sure:
  - Postgres has pgvector extension installed.
  - DB user in DSN can CREATE EXTENSION and create tables/indexes (or an admin can run those once).
"""

from __future__ import annotations
import asyncio
import hashlib
import json
from typing import List, Dict, Any, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import asyncpg

# -------------------------
# Config
# -------------------------
DB_DSN = "postgresql://username:password@localhost:5432/yourdb"  # <-- change me
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 10

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
VECTOR_MODEL = SentenceTransformer(MODEL_NAME)
VECTOR_DIM = VECTOR_MODEL.get_sentence_embedding_dimension()

PROFILE_VECTOR_SIZE = VECTOR_DIM
PRODUCT_VECTOR_SIZE = VECTOR_DIM * 3  # keep same concatenation approach

# HNSW index params recommendation (tweak for dataset)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

# -------------------------
# Utilities
# -------------------------
def string_to_point_id(input_string: str) -> int:
    """Deterministic integer id using SHA-256, reduced to 16 digits (0..10^16-1)."""
    return int(hashlib.sha256(input_string.encode("utf-8")).hexdigest(), 16) % (10 ** 16)


def get_text_embedding(text: str) -> np.ndarray:
    """Return numpy float32 embedding. Empty/invalid text returns zero vector."""
    if not text or not isinstance(text, str):
        return np.zeros(VECTOR_DIM, dtype=np.float32)
    emb = VECTOR_MODEL.encode(text, convert_to_tensor=False)
    emb_np = np.array(emb, dtype=np.float32)
    return emb_np


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit L2 norm safely (handles zero vectors)."""
    v = v.astype(np.float32)
    norm = np.linalg.norm(v)
    if norm <= 0:
        return v  # zero vector stays zero
    return (v / norm).astype(np.float32)


def _vec_to_pg_literal(v: np.ndarray) -> str:
    """
    Convert numpy array to Postgres vector literal string like '[0.1,0.2,...]'
    We cast it in SQL as '... '::vector.
    """
    # Use repr(float) to ensure e.g. 0.1 -> '0.1'
    return "[" + ",".join(map(lambda x: repr(float(x)), v.tolist())) + "]"


# -------------------------
# Vector builders (same logic as original but normalized)
# -------------------------
def build_profile_vector(page_view_keywords: List[str], purchase_keywords: List[str], interest_keywords: List[str]) -> Optional[np.ndarray]:
    if not page_view_keywords or not purchase_keywords or not interest_keywords:
        return None

    pv = np.mean([get_text_embedding(k) for k in page_view_keywords], axis=0)
    pu = np.mean([get_text_embedding(k) for k in purchase_keywords], axis=0)
    it = np.mean([get_text_embedding(k) for k in interest_keywords], axis=0)

    profile_vec = 0.3 * pv + 0.4 * pu + 0.3 * it
    profile_vec = normalize_vector(profile_vec)
    return profile_vec


def build_product_vector(product_name: str, product_category: str, product_keywords: List[str]) -> np.ndarray:
    name_v = get_text_embedding(product_name)
    cat_v = get_text_embedding(product_category)
    kws = np.array([get_text_embedding(k) for k in product_keywords]) if product_keywords else np.zeros((0, VECTOR_DIM), dtype=np.float32)
    kw_v = np.mean(kws, axis=0) if kws.shape[0] > 0 else np.zeros(VECTOR_DIM, dtype=np.float32)

    product_vec = np.concatenate([name_v, cat_v, kw_v]).astype(np.float32)
    product_vec = normalize_vector(product_vec)
    return product_vec


# -------------------------
# DB: pool and schema helpers
# -------------------------
async def create_pool(dsn: str = DB_DSN) -> asyncpg.pool.Pool:
    return await asyncpg.create_pool(dsn, min_size=POOL_MIN_SIZE, max_size=POOL_MAX_SIZE)


CREATE_PROFILES_SQL = f"""
CREATE TABLE IF NOT EXISTS profiles (
  id bigint PRIMARY KEY,
  profile_id text UNIQUE,
  embedding vector({PROFILE_VECTOR_SIZE}),
  payload jsonb
);
"""

CREATE_PRODUCTS_SQL = f"""
CREATE TABLE IF NOT EXISTS products (
  id bigint PRIMARY KEY,
  product_id text UNIQUE,
  embedding vector({PRODUCT_VECTOR_SIZE}),
  name text,
  category text,
  additional_info jsonb
);
"""

async def ensure_extension_and_tables(pool: asyncpg.pool.Pool) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            # create extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # create tables
            await conn.execute(CREATE_PROFILES_SQL)
            await conn.execute(CREATE_PRODUCTS_SQL)


async def create_hnsw_index_for_products(pool: asyncpg.pool.Pool, index_name: str = "products_embedding_hnsw"):
    """
    Create an HNSW index for products.embedding using recommended params.
    This can be expensive for big tables; run once when ready.
    """
    async with pool.acquire() as conn:
        # create index if not exists (pgvector supports IF NOT EXISTS for hnsw index)
        sql = f"""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = '{index_name}'
          ) THEN
            CREATE INDEX {index_name}
            ON products
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
          END IF;
        END$$;
        """
        await conn.execute(sql)


# -------------------------
# Upsert single
# -------------------------
async def upsert_profile(pool: asyncpg.pool.Pool, profile_id: str, page_view_keywords: List[str], purchase_keywords: List[str], interest_keywords: List[str], additional_info: Dict[str, Any]):
    profile_vec = build_profile_vector(page_view_keywords, purchase_keywords, interest_keywords)
    if profile_vec is None:
        raise ValueError("One or more keyword lists empty; cannot build profile vector.")

    id_int = string_to_point_id(profile_id)
    payload = {
        "profile_id": profile_id,
        "additional_info": additional_info,
        "page_view_keywords": page_view_keywords,
        "purchase_keywords": purchase_keywords,
        "interest_keywords": interest_keywords,
    }
    emb_literal = _vec_to_pg_literal(profile_vec)

    sql = """
    INSERT INTO profiles (id, profile_id, embedding, payload)
    VALUES ($1, $2, $3::vector, $4::jsonb)
    ON CONFLICT (id) DO UPDATE
      SET embedding = EXCLUDED.embedding,
          payload = EXCLUDED.payload,
          profile_id = EXCLUDED.profile_id;
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, id_int, profile_id, emb_literal, json.dumps(payload))
    return profile_id


async def upsert_product(pool: asyncpg.pool.Pool, product_id: str, product_name: str, product_category: str, product_keywords: List[str], additional_info: Dict[str, Any]):
    product_vec = build_product_vector(product_name, product_category, product_keywords)
    id_int = string_to_point_id(product_id)
    emb_literal = _vec_to_pg_literal(product_vec)

    sql = """
    INSERT INTO products (id, product_id, embedding, name, category, additional_info)
    VALUES ($1, $2, $3::vector, $4, $5, $6::jsonb)
    ON CONFLICT (id) DO UPDATE
      SET embedding = EXCLUDED.embedding,
          name = EXCLUDED.name,
          category = EXCLUDED.category,
          additional_info = EXCLUDED.additional_info,
          product_id = EXCLUDED.product_id;
    """

    async with pool.acquire() as conn:
        await conn.execute(sql, id_int, product_id, emb_literal, product_name, product_category, json.dumps(additional_info))
    return product_id


# -------------------------
# Batch upsert (profiles/products)
# Strategy:
# - create temporary table
# - executemany INSERT into temp table
# - single INSERT ... ON CONFLICT DO UPDATE from temp -> target table
# This is transactional and avoids many ON CONFLICT calls per row.
# -------------------------
async def batch_upsert_profiles(pool: asyncpg.pool.Pool, profiles: Sequence[Dict[str, Any]]):
    """
    profiles: list of dicts:
      {
        "profile_id": str,
        "page_view_keywords": [...],
        "purchase_keywords": [...],
        "interest_keywords": [...],
        "additional_info": {...}
      }
    """
    if not profiles:
        return 0

    async with pool.acquire() as conn:
        async with conn.transaction():
            # create temp table
            await conn.execute("""
              CREATE TEMP TABLE tmp_profiles (
                id bigint,
                profile_id text,
                embedding text,
                payload jsonb
              ) ON COMMIT DROP;
            """)
            # prepare rows for insertion into temp
            rows = []
            for p in profiles:
                pid = p["profile_id"]
                vec = build_profile_vector(p["page_view_keywords"], p["purchase_keywords"], p["interest_keywords"])
                if vec is None:
                    # skip invalid
                    continue
                rows.append((
                    string_to_point_id(pid),
                    pid,
                    _vec_to_pg_literal(vec),
                    json.dumps({
                        "profile_id": pid,
                        "additional_info": p.get("additional_info", {}),
                        "page_view_keywords": p["page_view_keywords"],
                        "purchase_keywords": p["purchase_keywords"],
                        "interest_keywords": p["interest_keywords"]
                    })
                ))

            # bulk insert into temp table
            await conn.executemany("INSERT INTO tmp_profiles (id, profile_id, embedding, payload) VALUES ($1, $2, $3, $4);", rows)

            # upsert into target table
            await conn.execute(f"""
              INSERT INTO profiles (id, profile_id, embedding, payload)
              SELECT id, profile_id, embedding::vector, payload FROM tmp_profiles
              ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    payload = EXCLUDED.payload,
                    profile_id = EXCLUDED.profile_id;
            """)
            return len(rows)


async def batch_upsert_products(pool: asyncpg.pool.Pool, products: Sequence[Dict[str, Any]]):
    """
    products: list of dicts:
      {
        "product_id": str,
        "name": str,
        "category": str,
        "keywords": [...],
        "additional_info": {...}
      }
    """
    if not products:
        return 0

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
              CREATE TEMP TABLE tmp_products (
                id bigint,
                product_id text,
                embedding text,
                name text,
                category text,
                additional_info jsonb
              ) ON COMMIT DROP;
            """)

            rows = []
            for p in products:
                pid = p["product_id"]
                vec = build_product_vector(p.get("name", ""), p.get("category", ""), p.get("keywords", []))
                rows.append((
                    string_to_point_id(pid),
                    pid,
                    _vec_to_pg_literal(vec),
                    p.get("name", ""),
                    p.get("category", ""),
                    json.dumps(p.get("additional_info", {}))
                ))

            await conn.executemany("INSERT INTO tmp_products (id, product_id, embedding, name, category, additional_info) VALUES ($1, $2, $3, $4, $5, $6);", rows)

            await conn.execute(f"""
              INSERT INTO products (id, product_id, embedding, name, category, additional_info)
              SELECT id, product_id, embedding::vector, name, category, additional_info FROM tmp_products
              ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    name = EXCLUDED.name,
                    category = EXCLUDED.category,
                    additional_info = EXCLUDED.additional_info,
                    product_id = EXCLUDED.product_id;
            """)
            return len(rows)


# -------------------------
# Recommendation search
# -------------------------
async def recommend_products_for_profile(pool: asyncpg.pool.Pool, profile_id: str, top_n: int = 8, except_product_ids: Optional[List[str]] = None):
    if except_product_ids is None:
        except_product_ids = []

    pid_int = string_to_point_id(profile_id)
    async with pool.acquire() as conn:
        rec = await conn.fetchrow("SELECT payload, embedding FROM profiles WHERE id = $1", pid_int)
        if not rec:
            return {"profile": None, "recommended_products": []}

        payload = rec["payload"]
        # prefer stored embedding if available
        embedding = rec["embedding"]
        if embedding is None:
            # rebuild from payload
            embedding = build_profile_vector(payload["page_view_keywords"], payload["purchase_keywords"], payload["interest_keywords"])
            if embedding is None:
                return {"profile": payload, "recommended_products": []}
        else:
            # asyncpg returns the pgvector as a list of floats
            embedding = np.array(embedding, dtype=np.float32)
        # triple-concat to match product vector size
        query_vec = np.concatenate([embedding, embedding, embedding]).astype(np.float32)
        query_literal = _vec_to_pg_literal(query_vec)

        # build exclusion clause
        exclude_clause = ""
        params: List[Any] = [query_literal]
        if except_product_ids:
            exclude_clause = "AND product_id NOT IN (" + ",".join(f"${i}" for i in range(2, 2 + len(except_product_ids))) + ")"
            params.extend(except_product_ids)
        params.append(top_n)  # last param

        sql = f"""
          SELECT product_id, name, category, additional_info,
                 1 - (embedding <=> $1::vector) AS score
          FROM products
          WHERE TRUE
          {exclude_clause}
          ORDER BY score DESC
          LIMIT ${len(params)};
        """
        rows = await conn.fetch(sql, *params)
        recommended = []
        for r in rows:
            recommended.append({
                "product_id": r["product_id"],
                "product_name": r["name"],
                "product_category": r["category"],
                "additional_info": r["additional_info"],
                "score": float(r["score"]) if r["score"] is not None else None
            })

        return {"profile": payload, "recommended_products": recommended}


# -------------------------
# Example usage
# -------------------------
async def main_demo():
    pool = await create_pool()
    try:
        await ensure_extension_and_tables(pool)
        # optionally create hnsw index (run once)
        await create_hnsw_index_for_products(pool)

        # Single upsert examples
        await upsert_profile(
            pool,
            profile_id="user_123",
            page_view_keywords=["running shoes", "trail shoes", "sports footwear"],
            purchase_keywords=["nike air running shoes"],
            interest_keywords=["trail running", "fitness"],
            additional_info={"age": 30}
        )

        await upsert_product(
            pool,
            product_id="sku_1001",
            product_name="Trailblazer Running Shoe",
            product_category="shoes",
            product_keywords=["trail", "grip", "lightweight"],
            additional_info={"brand": "BrandA", "price": 149.99}
        )

        await upsert_product(
            pool,
            product_id="sku_1002",
            product_name="City Runner Sneaker",
            product_category="shoes",
            product_keywords=["city", "cushion", "casual"],
            additional_info={"brand": "BrandB", "price": 109.99}
        )

        # Batch example
        profiles_batch = [
            {
                "profile_id": "user_batch_1",
                "page_view_keywords": ["bike helmet", "cycling shoes"],
                "purchase_keywords": ["helmet pro"],
                "interest_keywords": ["cycling"],
                "additional_info": {"country": "VN"}
            },
        ]
        await batch_upsert_profiles(pool, profiles_batch)

        products_batch = [
            {
                "product_id": "sku_2001",
                "name": "Mountain Grip Shoe",
                "category": "shoes",
                "keywords": ["mountain", "grippy", "durable"],
                "additional_info": {"brand": "BrandC", "price": 189.0}
            },
        ]
        await batch_upsert_products(pool, products_batch)

        # Recommend
        res = await recommend_products_for_profile(pool, "user_123", top_n=5)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main_demo())
