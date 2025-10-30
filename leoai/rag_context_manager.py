# leoai/rag_context_manager.py
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from leoai.db_utils import get_pg_conn, get_async_pg_conn, to_pgvector
from leoai.rag_db_manager import ChatDBManager
from leoai.rag_prompt_builder import PromptBuilder

logger = logging.getLogger("ContextManager")
DELTA_TO_REFRESH_CONTEXT = timedelta(seconds=10)

SUMMARY_PROMPT_TEMPLATE = """
You are a data extractor. Please analyze the conversation below. Extract key information and return a single, valid JSON object
enclosed in ```json ... ``` markdown block. Do not add any text before or after the JSON block.

Your JSON object MUST have this exact structure:
{{
  "user_profile": {{
    "first_name": "string or null",
    "last_name": "string or null",
    "primary_language": "string or null",
    "primary_email": "string or null",
    "primary_phone": "string or null",
    "personal_interests": ["list of strings"],
    "personality_traits": ["list of strings"],
    "data_labels": ["list of strings"],
    "in_segments": ["list of strings"],
    "in_journey_maps": ["list of strings"],
    "product_interests": ["list of strings"],
    "content_interests": ["list of strings"]
  }},
  "user_context": {{
    "location": "string or null",
    "datetime": "{now_str}"
  }},
  "context_summary": "the summary of long conversation",
  "context_keywords": ["list of keywords from long conversation"],
  "intent_label": "the label of intent from long conversation",
  "intent_confidence": "a probability score between 0 and 1"
}}

--- Conversation ---
{context}
"""


class ContextManager:
    def __init__(self, embedding_model, gemini_client, db_manager: ChatDBManager):
        self.embedding_model = embedding_model
        self.client = gemini_client
        self.db = db_manager

    async def build_context_summary(self, user_id, touchpoint_id, cdp_profile_id, user_message):
        current_context = self.get_context_summary(user_id, touchpoint_id)
        if self._needs_refresh(current_context):
            text_context = await self._retrieve_semantic_context(user_id, user_message)
            return await self._summarize_context(user_id, touchpoint_id, cdp_profile_id, text_context)
        return current_context

    def _needs_refresh(self, context):
        if not context:
            return True
        updated = context.get("updated_at")
        if not updated:
            return True
        now = datetime.now(timezone.utc)
        return (now - updated) > DELTA_TO_REFRESH_CONTEXT

    async def _summarize_context(self, user_id, touchpoint_id, cdp_profile_id, context):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not context:
            return {"user_profile": {}, "user_context": {"datetime": now_str}, "context_keywords": []}
        prompt = SUMMARY_PROMPT_TEMPLATE.format(context=context, now_str=now_str)
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: self.client.generate_content(prompt))
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if not match:
            logger.warning("No valid JSON block in summary output")
            return {"user_profile": {}, "user_context": {"datetime": now_str}, "context_keywords": []}
        try:
            summary = json.loads(match.group(1))
        except Exception:
            return {"user_profile": {}, "user_context": {"datetime": now_str}, "context_keywords": []}
        await self.db.save_context_summary(user_id, touchpoint_id, cdp_profile_id, summary)
        return summary

    async def _retrieve_semantic_context(self, user_id, user_message, limit=50):
        """Retrieve semantically similar messages."""
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(
                f"user: {user_message}", normalize_embeddings=True
            ).tolist()
        )
        vector_str = to_pgvector(vector)  # ✅ convert to pgvector format

        async with get_async_pg_conn() as conn:
            rows = await conn.fetch("""
                SELECT cm.message
                FROM chat_messages AS cm
                JOIN chat_message_embeddings AS ce
                ON cm.message_hash = ce.message_hash
                WHERE cm.user_id = $1
                ORDER BY ce.embedding <#> ($2)::vector ASC
                LIMIT $3;
            """, user_id, vector_str, limit)

        return "\n".join(r["message"] for r in rows) if rows else ""

    def get_context_summary(self, user_id, touchpoint_id):
        """Load the last saved context from DB."""
        with get_pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT context_data, updated_at FROM conversational_context
                WHERE user_id=%s AND touchpoint_id=%s;
            """, (user_id, touchpoint_id))
            row = cur.fetchone()
            if not row:
                return None
            context_data, updated_at = row
            if isinstance(context_data, str):
                context_data = json.loads(context_data)
            context_data["updated_at"] = updated_at
            return context_data
