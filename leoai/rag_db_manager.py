import asyncio
import json
import logging
from leoai.db_utils import get_async_pg_conn, sha256_hash, to_pgvector

logger = logging.getLogger("ChatDB")

class ChatDBManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def save_chat_message(self, user_id, role, message,
                                cdp_profile_id="_", persona_id="_",
                                touchpoint_id="_", keywords=[],
                                tenant_id="default"):
        if not user_id or not message:
            return

        if cdp_profile_id is None:
            cdp_profile_id = "_"
        if persona_id is None:
            persona_id = "_"
        if touchpoint_id is None:
            touchpoint_id = "_"
        if keywords is None:
            keywords = []

        msg_hash = sha256_hash(f"{user_id}:{message}")
        loop = asyncio.get_event_loop()

        msg_vector = await loop.run_in_executor(
            None,
            lambda: self.embedding_model.encode(
                f"{role}: {message}", normalize_embeddings=True
            ).tolist()
        )
        msg_vector_str = to_pgvector(msg_vector)  

        async with get_async_pg_conn() as conn:
            inserted = await conn.fetchrow("""
                INSERT INTO chat_messages
                (message_hash, user_id, cdp_profile_id, tenant_id, persona_id, touchpoint_id, role, message, keywords, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())
                ON CONFLICT (message_hash) DO NOTHING
                RETURNING message_hash;
            """, msg_hash, user_id, cdp_profile_id, tenant_id, persona_id, touchpoint_id, role, message, keywords)

            if inserted:
                await conn.execute("""
                    INSERT INTO chat_message_embeddings
                    (message_hash, tenant_id, embedding, created_at)
                    VALUES ($1,$2,$3::vector,NOW())
                    ON CONFLICT (message_hash) DO NOTHING;
                """, msg_hash, tenant_id, msg_vector_str)

        logger.info(f"💾 Stored {role} message for user={user_id}")

    async def save_context_summary(self, user_id: str,
                                   touchpoint_id: str,
                                   cdp_profile_id: str,
                                   summary: dict,
                                   tenant_id: str = "default") -> bool:
        """Save or update a conversational context summary (with pgvector embedding)."""
        
        if cdp_profile_id is None:
            cdp_profile_id = "_"
        if touchpoint_id is None:
            touchpoint_id = "_"

        try:
            context_json = json.dumps(summary, ensure_ascii=False)
            loop = asyncio.get_event_loop()

            # Create embedding in background thread
            embedding_vector = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(
                    context_json, normalize_embeddings=True
                ).tolist()
            )
            embedding_str = to_pgvector(embedding_vector)

            async with get_async_pg_conn() as conn:
                await conn.execute("""
                    INSERT INTO conversational_context (
                        user_id, touchpoint_id, cdp_profile_id, tenant_id,
                        context_data, embedding, intent_label, intent_confidence, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::vector,
                            COALESCE($7, ''), COALESCE($8, 0.0), NOW())
                    ON CONFLICT (user_id, touchpoint_id)
                    DO UPDATE SET
                        cdp_profile_id = EXCLUDED.cdp_profile_id,
                        context_data = EXCLUDED.context_data,
                        embedding = EXCLUDED.embedding,
                        intent_label = EXCLUDED.intent_label,
                        intent_confidence = EXCLUDED.intent_confidence,
                        updated_at = NOW();
                """,
                user_id,
                touchpoint_id,
                cdp_profile_id,
                tenant_id,
                context_json,
                embedding_str,
                summary.get("intent_label"),
                summary.get("intent_confidence", 0.0)
                )

            logger.info(f"💾 Context summary saved for user={user_id}, touchpoint={touchpoint_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save context summary: {e}")
            return False