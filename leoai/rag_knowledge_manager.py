# leoai/rag_knowledge_manager.py
import asyncio
import logging
from leoai.db_utils import get_async_pg_conn
from leoai.ai_knowledge_manager import MAX_DOC_TEXT_LENGTH

logger = logging.getLogger("KnowledgeRetriever")

class KnowledgeRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def retrieve(self, user_message: str, tenant_id: str, limit: int = 5):
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(user_message, normalize_embeddings=True).tolist()
        )

        async with await get_async_pg_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT kc.content FROM knowledge_chunks AS kc
                    JOIN knowledge_sources AS ks ON kc.source_id = ks.id
                    WHERE ks.tenant_id=%s AND ks.status='active'
                    ORDER BY kc.embedding <#> (%s)::vector ASC LIMIT %s;
                """, (tenant_id, vector, limit))
                rows = await cur.fetchall()
        if not rows:
            logger.info("No related knowledge found.")
            return ""
        chunks = [r[0].strip() for r in rows if r[0]]
        text = "\n\n---\n\n".join(chunks)
        return text[:MAX_DOC_TEXT_LENGTH]
