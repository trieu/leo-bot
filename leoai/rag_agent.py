import os
import hashlib
import logging
import markdown
import json
import re
from typing import Any, Dict, Optional, List, Union
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import asyncio

from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.db_utils import get_pg_conn, sha256_hash, get_async_pg_conn
from leoai.ai_knowledge_manager import MAX_DOC_TEXT_LENGTH
from main_config import REDIS_CLIENT

# --- Load environment variables from .env ---
load_dotenv(override=True)

# --- Configuration constants ---
TEMPERATURE_SCORE = float(os.getenv("TEMPERATURE_SCORE", 0.86))
VECTOR_DIMENSION = 768
MAX_CONTEXT_LENGTH = 10000
MAX_SUMMARY_LENGTH = 900
DELTA_TO_REFRESH_CONTEXT = timedelta(seconds=10)

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAGAgent")

# PROMPT
PROMPT_TEMPLATE = """Your name is LEO, a helpful and truthful AI assistant.
You must always respond in {target_language}.
If the context provided is insufficient to answer the user's question, state that clearly.

[Current Date and Time]
{datetime}

[User Profile Summary]
{user_profile}

[Conversation Context Summary]
{user_context}

[Key Conversation Summary]
{context_summary}

[Key Conversation Keywords]
{context_keywords}

---
[User's Current Question]
{question}
"""

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

def get_date_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_base_context(request_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
        get base context model
    """
    now_str = get_date_time_now()
    base = {
        "user_profile": {
            "first_name": None, "last_name": None,
            "primary_language": None, "primary_email": None, "primary_phone": None,
            "personal_interests": [], "personality_traits": [],
            "data_labels": [], "in_segments": [], "in_journey_maps": [],
            "product_interests": [], "content_interests": []
        },
        "user_context": {"location": None, "datetime": now_str},
        "context_summary": "",
        "context_keywords": [],
        "intent_label": None,
        "intent_confidence": 0.0
    }
    
    if request_data:
        def deep_merge(target: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in src.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        base = deep_merge(base, request_data)
    
    return base

# =====================================================================
# Main RAG pipeline
# =====================================================================

async def process_chat_message(
    user_id: str,
    user_message: str,
    cdp_profile_id: Optional[str] = None,
    persona_id: Optional[str] = None,
    touchpoint_id: Optional[str] = None,
    target_language: str = "Vietnamese",
    answer_in_format: str = "text",
    temperature_score: float = TEMPERATURE_SCORE,
    keywords: Optional[List[str]] = None,
    gemini_client: Optional[Union[GeminiClient, Any]] = None
) -> str:
    """
    Instantiates and runs the RAGAgent's async process.
    """
    agent = RAGAgent(gemini_client=gemini_client)
    return await agent.process_chat_message(
        user_id=user_id,
        user_message=user_message,
        cdp_profile_id=cdp_profile_id,
        persona_id=persona_id,
        touchpoint_id=touchpoint_id,
        target_language=target_language,
        answer_in_format=answer_in_format,
        temperature_score=temperature_score,
        keywords=keywords,
    )

class RAGAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to answer user queries.
    """
    def __init__(self, gemini_client: Optional[Union[GeminiClient, Any]] = None):
        self.client = gemini_client or GeminiClient()
        self.embedding_model = get_embedding_model()



    def _get_default_summary(self, timestamp_str: str) -> Dict[str, Any]:
        """Returns a default dictionary for summarization failures or short contexts."""
        return {
            "user_profile": {},
            "user_context": {"datetime": timestamp_str},
            "context_keywords": []
        }

    async def _summarize_context(
        self,
        user_id: str,
        touchpoint_id: str,
        cdp_profile_id: str,
        context: str,
        max_context_length: int = 10,
        temperature_score: float = 0.8
    ) -> Dict[str, Any]:
        """Summarizes long context into structured JSON metadata."""
        now_str = get_date_time_now()

        if not context:
            return self._get_default_summary(now_str)

        if len(context) < max_context_length:
            summary = self._get_default_summary(now_str)
            summary["user_context"]["raw_text"] = context
            logger.info("↔️ Context is short, skipping summarization.")
            return summary

        logging.info(f"🧩 Summarizing {len(context)} chars of context into structured JSON.")

        try:
            prompt = SUMMARY_PROMPT_TEMPLATE.format(now_str=now_str, context=context)
            # summary in event loop thread
            loop = asyncio.get_event_loop()
            raw_output = await loop.run_in_executor(None, lambda: self.client.generate_content(prompt, temperature=temperature_score))
            logging.info(f"🤖 Gemini output:\n{raw_output}")

            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
            if not match:
                logging.warning("⚠️ No valid JSON markdown block detected in Gemini summary output.")
                return self._get_default_summary(now_str)

            try:
                summary_json = json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logging.error(f"❌ JSON decoding error in summary: {e}\nRaw JSON text: {match.group(1)}")
                return self._get_default_summary(now_str)

            default_summary = self._get_default_summary(now_str)
            summary_json["user_profile"] = {**default_summary["user_profile"], **summary_json.get("user_profile", {})}
            summary_json["user_context"] = {**default_summary["user_context"], **summary_json.get("user_context", {})}
            summary_json.setdefault("context_keywords", [])
            summary_json.setdefault("context_summary", "")
            summary_json.setdefault("intent_label", None)
            # Ensure confidence is a float
            summary_json["intent_confidence"] = float(summary_json.get("intent_confidence", 0.0) or 0.0)

            logging.info("✅ Context successfully summarized into structured JSON.")
            
            await self._save_context_summary(user_id, touchpoint_id, cdp_profile_id, summary_json["intent_label"], summary_json["intent_confidence"], summary_json)
            return summary_json

        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON decoding error: {e}")
            return self._get_default_summary(now_str)
        except Exception as e:
            logging.error(f"❌ Summarization error: {e}")
            return self._get_default_summary(now_str)

    def _build_contextual_prompt(
        self,
        question: str,
        context_model: Dict[str, Any],
        target_language: str
    ) -> str:
        """Constructs a context-rich prompt for Gemini using the structured summary."""
        user_context = context_model.get("user_context", {})
        timestamp = user_context.get("datetime", get_date_time_now())
        try:
            if timestamp is None:
                timestamp = get_date_time_now()
            dt_object = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            current_time_str = dt_object.strftime("%A, %B %d, %Y at %I:%M %p")
        except ValueError:
            current_time_str = "Timestamp not available"

        user_profile_str = json.dumps(context_model.get("user_profile", {}), ensure_ascii=False, indent=2)
        user_context_str = json.dumps(user_context, ensure_ascii=False, indent=2)
        context_summary_str = context_model.get("context_summary", "")
        context_keywords_str = ", ".join(context_model.get("context_keywords", [])) or "None"

        return PROMPT_TEMPLATE.format(
            target_language=target_language,
            datetime=current_time_str,
            user_profile=user_profile_str,
            user_context=user_context_str,
            context_summary=context_summary_str,
            context_keywords=context_keywords_str,
            question=question.strip()
        )

    async def _save_chat_message(
        self,
        user_id: str,
        role: str,
        message: str,
        cdp_profile_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        touchpoint_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tenant_id: Optional[str] = "default"
    ):
        """Store a chat message and its embedding in the database."""
        
        if len(user_id) == 0 or len(message) == 0:
            # skip 
            return
        # Compute message hash
        message_hash = sha256_hash(f"{user_id}:{message}")
        
        # Run embedding computation in a background thread
        loop = asyncio.get_event_loop()
        message_vector = await loop.run_in_executor(
            None,
            lambda: self.embedding_model.encode(
                f"{role}: {message}", normalize_embeddings=True
            ).tolist()
        )

        async with await get_async_pg_conn() as conn:
            async with conn.cursor() as cur:
                # Insert message atomically
                await cur.execute("""
                    INSERT INTO chat_messages
                    (message_hash, user_id, cdp_profile_id, tenant_id, persona_id, touchpoint_id, role, message, keywords, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (message_hash) DO NOTHING
                    RETURNING message_hash
                """, (
                    message_hash, user_id, cdp_profile_id, tenant_id, persona_id,
                    touchpoint_id, role, message, keywords
                ))
                
                inserted = await cur.fetchone()
                if not inserted:
                    logger.debug(f"Duplicate message ignored (user={user_id}, role={role})")
                    return  # Duplicate, skip embedding

                # Insert embedding atomically
                await cur.execute("""
                    INSERT INTO chat_message_embeddings
                    (message_hash, tenant_id, embedding, created_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (message_hash) DO NOTHING
                """, (
                    message_hash, tenant_id, message_vector
                ))

            await conn.commit()

        logger.info(f"💾 Stored {role} message for user={user_id}")


    async def _retrieve_semantic_context(
        self,
        user_id: str,
        user_message: str,
        limit: int = 50,
        max_length: int = MAX_CONTEXT_LENGTH,
        tenant_id: Optional[str] = "default"
    ) -> str:
        """Retrieve semantically similar past chat messages for a given user and tenant."""
        loop = asyncio.get_event_loop()

        # Encode asynchronously to avoid blocking the event loop
        user_message_vector = await loop.run_in_executor(
            None,
            lambda: self.embedding_model.encode(
                f"user: {user_message}", normalize_embeddings=True
            ).tolist()
        )

        async with await get_async_pg_conn() as conn:
            async with conn.cursor() as cur:
                # Retrieve most semantically similar messages for this tenant and user
                await cur.execute("""
                    SELECT cm.message, (cme.embedding <#> (%s)::vector) AS distance
                    FROM chat_message_embeddings AS cme
                    JOIN chat_messages AS cm ON cm.message_hash = cme.message_hash
                    WHERE cme.tenant_id = %s
                    AND cm.user_id = %s
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (user_message_vector, tenant_id, user_id, limit))
                
                rows = await cur.fetchall()

        messages = [row[0] for row in rows] if rows else []
        if not messages:
            logger.info("🔍 No related chat history found.")
            return ""

        # Deduplicate messages while preserving order
        seen = set()
        unique_msgs = []
        for msg in messages:
            clean_msg = msg.strip()
            if clean_msg and clean_msg not in seen:
                seen.add(clean_msg)
                unique_msgs.append(clean_msg)

        full_context = "\n".join(unique_msgs)

        if len(full_context) > max_length:
            logger.warning(f"⚠️ Context too long ({len(full_context)} chars). Truncating.")
            return full_context[:max_length]

        logger.info(f"🧠 Retrieved {len(unique_msgs)} semantically similar messages for user={user_id}")
        return full_context


    async def _save_context_summary(
        self,
        user_id: str,
        touchpoint_id: str,
        cdp_profile_id: str,
        intent_label: str,
        intent_confidence: float,
        context_data: Dict[str, Any]
    ) -> bool:
        """Save or update a conversational context record with embeddings."""
        try:
            loop = asyncio.get_event_loop()
            context_json = json.dumps(context_data)
            context_embedding = await loop.run_in_executor(None, lambda: self.embedding_model.encode(context_json, normalize_embeddings=True).tolist())

            async with await get_async_pg_conn() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                    INSERT INTO conversational_context (
                        user_id, touchpoint_id, cdp_profile_id, context_data, embedding, intent_label, intent_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, touchpoint_id)
                    DO UPDATE SET
                        cdp_profile_id = EXCLUDED.cdp_profile_id,
                        context_data = EXCLUDED.context_data,
                        embedding = EXCLUDED.embedding,
                        intent_label = EXCLUDED.intent_label,
                        intent_confidence = EXCLUDED.intent_confidence,
                        updated_at = NOW();
                """, (user_id, touchpoint_id, cdp_profile_id, context_json, context_embedding, intent_label, intent_confidence))
                await conn.commit()

            logger.info(f"💾 Context summary saved for user={user_id}, touchpoint={touchpoint_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save context summary: {e}")
            return False

    def _fetch_context_from_db(self, user_id: str, touchpoint_id: str) -> Optional[tuple]:
        """Fetches the context summary row from the database."""
        with get_pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT context_data, intent_label, intent_confidence, updated_at
                FROM conversational_context
                WHERE user_id = %s AND touchpoint_id = %s;
            """, (user_id, touchpoint_id))
            return cur.fetchone()



    def _parse_db_row_to_context(self, row: tuple) -> Dict[str, Any]:
        """Parses the database row and merges it into a base context structure."""
        base_context = get_base_context()
        context_data, intent_label, intent_confidence, updated_at = row

        if isinstance(context_data, str):
            try:
                context_data = json.loads(context_data)
            except json.JSONDecodeError:
                context_data = {}
        elif not isinstance(context_data, dict):
            context_data = {}

        # Merge data into the base structure
        base_context["user_profile"].update(context_data.get("user_profile", {}))
        base_context["user_context"].update(context_data.get("user_context", {}))
        base_context["context_summary"] = context_data.get("context_summary", "")
        base_context["context_keywords"] = context_data.get("context_keywords", [])
        base_context["updated_at"] = updated_at
        base_context["intent_label"] = intent_label
        base_context["intent_confidence"] = float(intent_confidence or 0.0)

        return base_context

    def _get_context_summary(self, user_id: str, touchpoint_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and normalize a structured conversational context summary."""
        try:
            row = self._fetch_context_from_db(user_id, touchpoint_id)

            if not row:
                logger.info(f"ℹ️ No context summary found for user={user_id}, touchpoint={touchpoint_id}")
                return None

            context = self._parse_db_row_to_context(row)

            logger.info(f"ℹ️ Found context summary for user={user_id}, touchpoint={touchpoint_id} intent_label={context['intent_label']}")
            return context

        except Exception as e:
            logger.error(f"❌ Failed to load context summary for user={user_id}, touchpoint={touchpoint_id}: {e}")
            return None

    async def _build_context_summary(self, user_id: str, touchpoint_id: str, cdp_profile_id: str,  user_message: str) -> Dict[str, Any]:
        logger.info(f"🧠 build_context_summary for user_id: {user_id} touchpoint_id: {touchpoint_id}")
        summarized_context = self._get_context_summary(user_id, touchpoint_id)

        needs_refresh = self._check_for_refresh_context(summarized_context)

        if needs_refresh:
            logger.info("🔄 Refreshing summarized context...")
            retrieved_context_str = await self._retrieve_semantic_context(user_id=user_id, user_message=user_message)
            summarized_context = await self._summarize_context(
                user_id,
                touchpoint_id,
                cdp_profile_id,
                context=retrieved_context_str
            ) 

        return summarized_context or self._get_default_summary(get_date_time_now())

    def _check_for_refresh_context(self, summarized_context):
        needs_refresh = True
        if summarized_context:
            updated_at = summarized_context.get("updated_at")
            if updated_at:
                now_utc = datetime.now(timezone.utc)
                if (now_utc - updated_at) < DELTA_TO_REFRESH_CONTEXT:
                    needs_refresh = False
        return needs_refresh
    
    async def _retrieve_knowledge(
        self,
        user_message: str,
        tenant_id: str,
        limit: int = 5,
        max_length: int = MAX_DOC_TEXT_LENGTH
    ) -> str:
        """
        Retrieve semantically relevant knowledge chunks for a given user message and tenant.
        """
        loop = asyncio.get_event_loop()

        # 1. Encode the user's question into a vector
        try:
            user_message_vector = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(
                    user_message, normalize_embeddings=True
                ).tolist()
            )
        except Exception as e:
            logger.error(f"❌ Failed to encode user message for knowledge retrieval: {e}")
            return ""

        # 2. Query the database for the most similar chunks
        async with await get_async_pg_conn() as conn:
            async with conn.cursor() as cur:
                # Retrieve chunks from active knowledge sources for this tenant
                await cur.execute("""
                    SELECT
                        kc.content,
                        (kc.embedding <#> (%s)::vector) AS distance
                    FROM knowledge_chunks AS kc
                    JOIN knowledge_sources AS ks ON kc.source_id = ks.id
                    WHERE
                        ks.tenant_id = %s
                        AND ks.status = 'active'
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (user_message_vector, tenant_id, limit))
                
                rows = await cur.fetchall()

        chunks = [row[0] for row in rows] if rows else []
        if not chunks:
            logger.info("🔍 No related knowledge found in the database.")
            return ""

        # 3. Process and format the results
        # Deduplicate chunks while preserving order of relevance
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            clean_chunk = chunk.strip()
            if clean_chunk and clean_chunk not in seen:
                seen.add(clean_chunk)
                unique_chunks.append(clean_chunk)

        full_context = "\n\n---\n\n".join(unique_chunks)

        if len(full_context) > max_length:
            logger.warning(f"⚠️ Retrieved knowledge context is too long ({len(full_context)} chars). Truncating.")
            return full_context[:max_length]

        logger.info(f"🧠 Retrieved {len(unique_chunks)} semantically similar knowledge chunks for tenant={tenant_id}")
        return full_context

    async def process_chat_message(
        self,
        user_id: str,
        user_message: str,
        cdp_profile_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        touchpoint_id: Optional[str] = None,
        target_language: str = "Vietnamese",
        answer_in_format: str = "text",
        temperature_score: float = TEMPERATURE_SCORE,
        keywords: Optional[List[str]] = None,
    ) -> str:
        """
        End-to-end RAG process:
          1. Save user message.
          2. Build summarized context.
          3. Build contextual prompt.
          4. Generate AI answer.
          5. Save AI answer.
          6. Return formatted answer.
        """
        try:
            # 1 Save user message.
            await self._save_chat_message(user_id, "user", user_message, cdp_profile_id, persona_id, touchpoint_id, keywords)

            # 2 Build summarized context.
            summarized_context = await self._build_context_summary(user_id, touchpoint_id, cdp_profile_id, user_message)
            
            #  TODO Update Redis with the user's first name for quick access
            if first_name := summarized_context.get("user_profile", {}).get("first_name"):
                REDIS_CLIENT.hset(user_id, mapping={"profile_id": "", "name": first_name})

            # 3 Build contextual prompt.
            final_prompt = self._build_contextual_prompt(user_message, summarized_context, target_language)
            logger.info(f"🧠 Final Prompt:\n{final_prompt}")

            # 4 Generate AI answer.
            answer = self.client.generate_content(final_prompt, temperature=temperature_score)
            if not answer:
                return "⚠️ I couldn't find enough information to answer that confidently."

            # 5 Save AI answer.
            await self._save_chat_message(user_id, 'bot', answer, 'bot', persona_id, touchpoint_id)

            # 6 Return formatted answer.
            return markdown.markdown(answer) if answer_in_format == "html" else answer

        except Exception as e:
            logger.exception("❌ RAG pipeline error")
            # A more generic, safer error message for the end-user.
            return "I'm sorry, but I encountered an unexpected error and can't process your request right now. Please try again later."
        