# app.py
"""
FastAPI RAG service using PostgreSQL 16 + pgvector for retrieval
and Mistral-7B-Instruct (GGUF via llama.cpp) for generation.

Added:
- Multi-tenant RBAC using API keys (viewer/editor/admin)
- Tenant isolation for endpoints
- Streaming /ask/stream with SSE
"""

import asyncio
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Dict, Any, AsyncGenerator

import asyncpg
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
import numpy as np
from llama_cpp import Llama
from contextlib import asynccontextmanager

# -------------------- Config --------------------
@dataclass
class Settings:
    pg_dsn: str = os.getenv("PG_DSN", "postgresql://rag_user:changeme@localhost:5432/customer360")
    model_embed: str = os.getenv("MODEL_EMBED", "intfloat/multilingual-e5-base")
    mistral_path: str = os.getenv("MISTRAL_GGUF", "./mistral-7b-instruct-v0.2.Q6_K.gguf")
    llama_ctx: int = int(os.getenv("LLAMA_CTX", "4096"))
    llama_threads: int = int(os.getenv("LLAMA_THREADS", "8"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _embedder, _llm
    # Startup
    _pool = await asyncpg.create_pool(
        dsn=settings.pg_dsn,
        min_size=1,
        max_size=8
    )
    await ensure_schema(_pool)

    _embedder = SentenceTransformer(settings.model_embed)
    _llm = Llama(
        model_path=settings.mistral_path,
        n_ctx=settings.llama_ctx,
        n_threads=settings.llama_threads,
        logits_all=False,
        verbose=False,
    )
    yield
    # Shutdown
    if _pool:
        await _pool.close()
    del _llm
    del _embedder

app = FastAPI(
    title="RAG API (pgvector + e5 + Mistral) with RBAC + SSE",
    lifespan=lifespan
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# -------------------- Globals --------------------
_pool: Optional[asyncpg.Pool] = None
_embedder: Optional[SentenceTransformer] = None
_llm: Optional[Llama] = None

# -------------------- Utils --------------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.model_embed)
    return _embedder

@lru_cache(maxsize=1)
def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=settings.mistral_path,
            n_ctx=settings.llama_ctx,
            n_threads=settings.llama_threads,
            logits_all=False,
            verbose=False,
        )
    return _llm

async def ensure_schema(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE SCHEMA IF NOT EXISTS rag;
            CREATE TABLE IF NOT EXISTS rag.documents (
                id           BIGSERIAL PRIMARY KEY,
                tenant_id    TEXT NOT NULL DEFAULT 'default',
                doc_id       TEXT NOT NULL,
                chunk_id     INT  NOT NULL,
                content      TEXT NOT NULL,
                metadata     JSONB NOT NULL DEFAULT '{}',
                embedding    vector(768) NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(tenant_id, doc_id, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_tenant ON rag.documents(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_documents_docid  ON rag.documents(tenant_id, doc_id);

            CREATE TABLE IF NOT EXISTS rag.api_keys (
                api_key TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('viewer','editor','admin')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )

# -------------------- Pydantic models --------------------
class IngestDoc(BaseModel):
    tenant_id: str = Field(default="default")
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    items: List[IngestDoc]
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class SearchRequest(BaseModel):
    tenant_id: str = Field(default="default")
    query: str
    top_k: Optional[int] = None

class AskRequest(BaseModel):
    tenant_id: str = Field(default="default")
    query: str
    top_k: Optional[int] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None

# -------------------- Auth / RBAC --------------------
async def get_api_key_info(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT api_key, tenant_id, role FROM rag.api_keys WHERE api_key=$1", x_api_key)
        if not row:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return {"api_key": row["api_key"], "tenant_id": row["tenant_id"], "role": row["role"]}

def check_tenant_allowed(request_tenant: str, key_info: Dict[str, Any]):
    if key_info["role"] == "admin":
        return True
    if key_info["tenant_id"] != request_tenant:
        raise HTTPException(status_code=403, detail="API key not authorized for this tenant")
    return True



# -------------------- Endpoints --------------------
@app.post("/ingest")
async def ingest(payload: IngestRequest, key_info: Dict[str, Any] = Depends(get_api_key_info)):
    if key_info["role"] not in ("editor", "admin"):
        raise HTTPException(status_code=403, detail="API key role cannot ingest")
    for item in payload.items:
        check_tenant_allowed(item.tenant_id, key_info)

    chunk_size = payload.chunk_size or settings.chunk_size
    chunk_overlap = payload.chunk_overlap or settings.chunk_overlap

    chunk_records = []
    for item in payload.items:
        chunks = chunk_text(item.text, chunk_size, chunk_overlap)
        for i, ch in enumerate(chunks):
            chunk_records.append((item.tenant_id, item.doc_id, i, ch, json.dumps(item.metadata)))

    if not chunk_records:
        return {"ingested": 0}

    loop = asyncio.get_event_loop()
    embedder = get_embedder()
    contents = [c[3] for c in chunk_records]
    vectors = await loop.run_in_executor(None, lambda: embedder.encode(contents, normalize_embeddings=True))
    vectors = np.asarray(vectors, dtype=np.float32)

    async with _pool.acquire() as conn:
        async with conn.transaction():
            stmt = """
            INSERT INTO rag.documents (tenant_id, doc_id, chunk_id, content, metadata, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (tenant_id, doc_id, chunk_id)
            DO UPDATE SET content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding
            """
            for (tenant_id, doc_id, chunk_id, content, metadata), vec in zip(chunk_records, vectors):
                await conn.execute(stmt, tenant_id, doc_id, chunk_id, content, metadata, vec.tolist())

    return {"ingested": len(chunk_records), "documents": len(payload.items)}

@app.post("/search")
async def search(payload: SearchRequest, key_info: Dict[str, Any] = Depends(get_api_key_info)):
    check_tenant_allowed(payload.tenant_id, key_info)
    top_k = payload.top_k or settings.top_k
    if not payload.query:
        raise HTTPException(400, detail="Empty query")

    loop = asyncio.get_event_loop()
    embedder = get_embedder()
    q_vec = await loop.run_in_executor(None, lambda: embedder.encode([payload.query], normalize_embeddings=True))
    q_vec = q_vec[0].tolist()

    sql = """
        SELECT doc_id, chunk_id, content, metadata, 1 - (embedding <=> $1::vector) AS score
        FROM rag.documents
        WHERE tenant_id = $2
        ORDER BY embedding <=> $1::vector ASC
        LIMIT $3
    """

    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, q_vec, payload.tenant_id, top_k)

    results = [
        {
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "content": r["content"],
            "metadata": r["metadata"],
            "score": float(r["score"]),
        }
        for r in rows
    ]
    return {"results": results}

SYSTEM_TEMPLATE = (
    "Bạn là trợ lý AI tên là TRIEU, trả lời bằng tiếng Việt, súc tích, chính xác. "
    "Chỉ sử dụng thông tin từ ngữ cảnh khi có, nếu thiếu hãy nói rõ giới hạn."
)

USER_TEMPLATE = (
    "Ngữ cảnh (các đoạn liên quan):\n{context}\n\n"
    "Câu hỏi: {query}\n"
    "Yêu cầu: Trả lời bằng tiếng Việt, có trích dẫn [doc_id#chunk_id] khi phù hợp."
)

def build_prompt(context_chunks: List[Dict[str, Any]], query: str, system_prompt: Optional[str]) -> str:
    system = system_prompt or SYSTEM_TEMPLATE
    ctx_lines = []
    for c in context_chunks:
        tag = f"[{c['doc_id']}#{c['chunk_id']}]"
        ctx_lines.append(f"{tag} {c['content']}")
    ctx = "\n\n".join(ctx_lines) if ctx_lines else "(không có)"
    prompt = """<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{usr} [/INST]""".format(
        sys=system,
        usr=USER_TEMPLATE.format(context=ctx, query=query),
    )
    return prompt

@app.post("/ask")
async def ask(payload: AskRequest, key_info: Dict[str, Any] = Depends(get_api_key_info)):
    check_tenant_allowed(payload.tenant_id, key_info)
    sr = await search(SearchRequest(tenant_id=payload.tenant_id, query=payload.query, top_k=payload.top_k), key_info)
    context_chunks = sr["results"]
    prompt = build_prompt(context_chunks, payload.query, payload.system_prompt)

    llm = get_llm()
    out = llm(
        prompt,
        max_tokens=payload.max_tokens or settings.max_output_tokens,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
    )
    text = out["choices"][0]["text"].strip()
    return {"answer": text, "context": context_chunks}

# -------------------- SSE streaming /ask/stream --------------------
async def _sentence_splitter(text: str) -> List[str]:
    import re
    parts = re.split(r'(?<=[。.!?\n])\s*', text)
    parts = [p for p in parts if p.strip()]
    return parts

@app.get("/ask/stream")
async def ask_stream(tenant_id: str, query: str, x_api_key: Optional[str] = Header(None)):
    key_info = await get_api_key_info(x_api_key)
    check_tenant_allowed(tenant_id, key_info)

    loop = asyncio.get_event_loop()
    embedder = get_embedder()
    q_vec = await loop.run_in_executor(None, lambda: embedder.encode([query], normalize_embeddings=True))
    q_vec = q_vec[0].tolist()

    sql = """
        SELECT doc_id, chunk_id, content, metadata, 1 - (embedding <=> $1::vector) AS score
        FROM rag.documents
        WHERE tenant_id = $2
        ORDER BY embedding <=> $1::vector ASC
        LIMIT $3
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, q_vec, tenant_id, settings.top_k)
    context_chunks = [
        {"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "content": r["content"], "metadata": r["metadata"], "score": float(r["score"])}
        for r in rows
    ]

    prompt = build_prompt(context_chunks, query, None)
    llm = get_llm()

    # generate full text (replace with streaming tokens if available)
    out = llm(prompt, max_tokens=settings.max_output_tokens, temperature=0.2)
    text = out["choices"][0]["text"].strip()
    pieces = await _sentence_splitter(text)

    async def event_generator() -> AsyncGenerator[bytes, None]:
        meta = {"type": "meta", "context": [{"doc_id": c['doc_id'], "chunk_id": c['chunk_id'], "score": c['score']} for c in context_chunks]}
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n".encode("utf-8")
        for i, piece in enumerate(pieces):
            data = {"type": "partial", "index": i, "text": piece}
            yield f"event: partial\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
            await asyncio.sleep(0.02)
        yield f"event: done\ndata: {{}}\n\n".encode("utf-8")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Admin helper
ADMIN_SQL_SNIPPET = """
-- create an API key for tenant 'acme' with role editor
INSERT INTO rag.api_keys(api_key, tenant_id, role) VALUES ('my-secret-key-1', 'acme', 'editor');
-- create admin key
INSERT INTO rag.api_keys(api_key, tenant_id, role) VALUES ('admin-key-000', 'super', 'admin');
"""

@app.get('/admin/sql-snippet')
async def admin_sql_snippet(key_info: Dict[str, Any] = Depends(get_api_key_info)):
    if key_info['role'] != 'admin':
        raise HTTPException(status_code=403, detail='admin role required')
    return JSONResponse({'sql': ADMIN_SQL_SNIPPET})
