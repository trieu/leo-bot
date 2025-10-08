import os
import hashlib

import logging
import markdown
import json
import re
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone


from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.db_utils import get_pg_conn

# --- Load environment variables from .env ---
load_dotenv(override=True)

# --- Configuration constants ---
TEMPERATURE_SCORE = float(os.getenv("TEMPERATURE_SCORE", 0.86))
VECTOR_DIMENSION = 768
MAX_CONTEXT_LENGTH = 5000
MAX_SUMMARY_LENGTH = 900


# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAGAgent")

# --- Initialize embedding model ---
embedding_model = get_embedding_model()

# =====================================================================
# Utility functions
# =====================================================================

def sha256_hash(text: str) -> str:
    """Generate a SHA256 hash of text to detect duplicate messages."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


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

def _get_default_summary(timestamp_str: str) -> Dict[str, Any]:
    """
    # IMPROVEMENT: DRY Principle. Centralizes the default return structure.
    Returns a default dictionary for summarization failures or short contexts.
    """
    return {
        "user_profile": {},
        "user_context": {"datetime": timestamp_str},
        "context_keywords": []
    }
    
def get_date_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def summarize_context(
    user_id: str,
    touchpoint_id: str,
    context: str,
    max_context_length: int = 10,
    temperature_score: float = 0.8,
    gemini_client: Optional[Any] = None # Using Any for placeholder
) -> Dict[str, Any]:
    """
    Summarizes long context into structured JSON metadata.
    Extracts user_profile, user_context, and context_keywords.
    """
    # IMPROVEMENT: Calculate datetime once at the start for consistency.
    now_str = get_date_time_now()

    if not context:
        return _get_default_summary(now_str)

    # --- BUG FIX: Correctly handle short contexts ---
    # The original code discarded the short context. This version preserves it.
    if len(context) < max_context_length:
        summary = _get_default_summary(now_str)
        summary["user_context"]["raw_text"] = context
        logging.info("↔️ Context is short, skipping summarization.")
        return summary

    logging.info(f"🧩 Summarizing {len(context)} chars of context into structured JSON.")

    try:
        client = gemini_client or GeminiClient()

        # IMPROVEMENT: More robust prompt asking for markdown code fences.
        prompt = f"""
        You are data extractor. Please analyze the conversation below. Extract key information and return a single, valid JSON object
        enclosed in ```json ... ``` markdown block. Do not add any text before or after the JSON block.

        Your JSON object MUST have this exact structure:
        {{
          "user_profile": {{
            "first_name": "string or null",
            "last_name": "string or null",
            "interests": ["list of strings"],
            "personality_traits": ["list of strings"]
          }},
          "user_context": {{
            "location": "string or null",
            "datetime": "{now_str}"
          }},
          "context_summary": "the summary of long conversation",
          "context_keywords": ["list of keywords from long conversation"],
          "intent_label":"the label of intent from long conversation",
          "intent_confidence": a probability score between 0 and 1
        }}

        --- Conversation ---
        {context}
        """

        raw_output = client.generate_content(prompt, temperature=temperature_score)
        logging.info(f"🤖 Gemini output:\n{raw_output}")

        # IMPROVEMENT: More robust regex to find a JSON block within markdown fences.
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if not match:
            logging.warning("⚠️ No valid JSON markdown block detected in Gemini summary output.")
            return _get_default_summary(now_str)

        summary_json = json.loads(match.group(1))

        # Ensure consistent structure using a more concise method.
        # Merging ensures default keys are present if the LLM omits them.
        default_summary = _get_default_summary(now_str)
        summary_json["user_profile"] = {**default_summary["user_profile"], **summary_json.get("user_profile", {})}
        summary_json["user_context"] = {**default_summary["user_context"], **summary_json.get("user_context", {})}
        summary_json.setdefault("context_keywords", [])
        summary_json.setdefault("context_summary", "")
        summary_json.setdefault("intent_label", "")
        summary_json.setdefault("intent_confidence", 0)

        logging.info("✅ Context successfully summarized into structured JSON.")
        
        save_context_summary(user_id, touchpoint_id, summary_json["intent_label"], summary_json["intent_confidence"], summary_json)
        return summary_json

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON decoding error: {e}")
        return _get_default_summary(now_str)
    except Exception as e:
        logging.error(f"❌ Summarization error: {e}")
        return _get_default_summary(now_str)

def _build_prompt(
    question: str,
    context_model: Dict[str, Any],
    target_language: str
) -> str:
    """
    Constructs a context-rich prompt for Gemini using the structured summary.
    """
    # IMPROVEMENT: Get datetime from the context summary for consistency.
    user_context = context_model.get("user_context", {})
    timestamp = user_context.get("datetime", get_date_time_now())
    try:
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


# =====================================================================
# Database operations
# =====================================================================

def save_chat_message(
    user_id: str,
    role: str,
    message: str,
    persona_id: Optional[str] = None,
    touchpoint_id: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    tenant_id: Optional[str] = "default"
):
    """
    Store a chat message and its embedding in the database.
    Avoids duplicates using a SHA256 hash check.
    """
    message_hash = sha256_hash(message)
    message_vector = embedding_model.encode(f"{role}: {message}", normalize_embeddings=True).tolist()

    with get_pg_conn() as conn, conn.cursor() as cur:
        # --- Check duplicate ---
        cur.execute("""
            SELECT 1 FROM chat_messages WHERE message_hash = %s AND user_id = %s
        """, (message_hash, user_id))
        if cur.fetchone():
            logger.debug(f"Duplicate message ignored (user={user_id}, role={role})")
            return

        # --- Insert message ---
        cur.execute("""
            INSERT INTO chat_messages
            (user_id, tenant_id, persona_id, touchpoint_id, role, message, message_hash, keywords, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
        """, (user_id, tenant_id, persona_id, touchpoint_id, role, message, message_hash, keywords))

        # --- Insert message embedding ---
        cur.execute("""
            INSERT INTO chat_history_embeddings
            (user_id, tenant_id, persona_id, touchpoint_id, role, message, keywords, embedding, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
        """, (user_id, tenant_id, persona_id, touchpoint_id, role, message, keywords, message_vector))

    logger.info(f"💾 Stored {role} message for user={user_id}")


# =====================================================================
# Retrieval and summarization
# =====================================================================

def retrieve_semantic_context(
    user_id: str, question: str, k: int = 20, max_length: int = MAX_CONTEXT_LENGTH
) -> str:
    """
    Retrieve semantically similar chat history entries.
    Uses vector similarity (<#> operator) and filters duplicates.
    """
    question_vector = embedding_model.encode(f"user: {question}", normalize_embeddings=True).tolist()

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT message, (embedding <#> (%s)::vector) AS distance
            FROM chat_history_embeddings
            WHERE user_id = %s
            ORDER BY distance ASC
            LIMIT %s
        """, (question_vector, user_id, k))
        rows = cur.fetchall()

    messages = [row[0] for row in rows]
    if not messages:
        logger.info("🔍 No related history found.")
        return ""

    # --- Deduplicate results to avoid redundant context ---
    seen = set()
    unique_msgs = []
    for msg in messages:
        clean_msg = msg.strip()
        if clean_msg not in seen:
            seen.add(clean_msg)
            unique_msgs.append(clean_msg)

    # --- Combine into a single context block ---
    full_context = "\n".join(unique_msgs)

    # --- Truncate if exceeds max_length ---
    if len(full_context) > max_length:
        logger.warning(f"Context too long ({len(full_context)} chars). Truncating.")
        return full_context[:max_length]

    return full_context


def save_context_summary(
    user_id: str,
    touchpoint_id: str,
    intent_label: str,
    intent_confidence: float,
    context_data: Dict[str, Any]
) -> bool:
    """
    Save or update a conversational context record with embeddings.
    """
    try:
        # Convert context JSON and embed it semantically for retrieval
        context_json = json.dumps(context_data)
        context_embedding = embedding_model.encode(context_json, normalize_embeddings=True).tolist()

        with get_pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversational_context (
                    user_id, touchpoint_id, context_data, embedding, intent_label, intent_confidence
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, touchpoint_id)
                DO UPDATE SET
                    context_data = EXCLUDED.context_data,
                    embedding = EXCLUDED.embedding,
                    intent_label = EXCLUDED.intent_label,
                    intent_confidence = EXCLUDED.intent_confidence,
                    updated_at = NOW();
            """, (user_id, touchpoint_id, context_json, context_embedding, intent_label, intent_confidence))
            conn.commit()

        logger.info(f"💾 Context summary saved for user={user_id}, touchpoint={touchpoint_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to save context summary: {e}")
        return False

def get_context_summary(user_id: str, touchpoint_id: str) -> Dict[str, Any]:
    """
    Retrieve and normalize a structured conversational context summary.
    Ensures consistent return schema even when data is missing or incomplete.
    """
    try:
        with get_pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT context_data, intent_label, intent_confidence, updated_at
                FROM conversational_context
                WHERE user_id = %s AND touchpoint_id = %s;
            """, (user_id, touchpoint_id))
            row = cur.fetchone()

        # Default empty structure
        now_str = get_date_time_now()
        base_context = {
            "user_profile": {
                "first_name": None,
                "last_name": None,
                "interests": [],
                "personality_traits": []
            },
            "user_context": {
                "location": None,
                "datetime": now_str
            },
            "context_summary": "",
            "context_keywords": [],
            "intent_label": None,
            "intent_confidence": 0.0
        }

        # If nothing in DB — return defaults
        if not row:
            logger.info(f"ℹ️ No context summary found for user={user_id}, touchpoint={touchpoint_id}")
            return None

        context_data, intent_label, intent_confidence , updated_at = row

        # Merge stored JSONB fields into expected schema
        if context_data:
            if isinstance(context_data, str):
                try:
                    context_data = json.loads(context_data)
                except json.JSONDecodeError:
                    context_data = {}

            base_context["user_profile"].update(context_data.get("user_profile", {}))
            base_context["user_context"].update(context_data.get("user_context", {}))
            base_context["context_summary"] = context_data.get("context_summary", "")
            base_context["context_keywords"] = context_data.get("context_keywords", [])
            base_context["context_keywords"] = context_data.get("context_keywords", [])
            base_context["updated_at"] = updated_at

        # Override with database-level intent info
        base_context["intent_label"] = intent_label
        base_context["intent_confidence"] = float(intent_confidence) if intent_confidence else 0.0

        logger.info(f"ℹ️ Found context summary found for user={user_id}, touchpoint={touchpoint_id} intent_label={intent_label}")
        return base_context

    except Exception as e:
        logger.error(f"❌ Failed to load context summary for user={user_id}, touchpoint={touchpoint_id}: {e}")
        # Return safe empty structure if anything fails
        now_str =  datetime.now().isoformat()
        return None


def build_context_summary(user_id: str, touchpoint_id: str, user_message: str, client: GeminiClient) -> Dict[str, Any]:
    logger.info(f"🧠 build_context_summary for user_id: {user_id} touchpoint_id: {touchpoint_id}")
    # --- Step 1: Get cached summarized_context for user ---
    summarized_context = get_context_summary(user_id, touchpoint_id)

    # Check if cache exists and is recent
    needs_refresh = True
    if summarized_context:
        updated_at = summarized_context.get("updated_at")
        if updated_at:
            # updated_at is already aware from Postgres, just in UTC
            now_utc = datetime.now(timezone.utc)  # aware UTC datetime
            time_diff = now_utc - updated_at
            if time_diff < timedelta(minutes=3):
                needs_refresh = False

    # --- Step 2: Refresh summary if needed ---
    if needs_refresh:
        logger.info("🔄 Refreshing summarized context...")

        # Retrieve related past messages (semantic recall)
        retrieved_context_str = retrieve_semantic_context(user_id, user_message)

        # Summarize using Gemini or LLM
        summarized_context = summarize_context(
            user_id,
            touchpoint_id,
            context=retrieved_context_str,
            gemini_client=client
        )

    return summarized_context


# =====================================================================
# Main RAG pipeline
# =====================================================================

def process_chat_message(
    user_id: str,
    user_message: str,
    persona_id: Optional[str] = None,
    touchpoint_id: Optional[str] = None,
    target_language: str = "Vietnamese",
    answer_in_format: str = "text",
    temperature_score: float = TEMPERATURE_SCORE,
    keywords: Optional[List[str]] = None,
    gemini_client: Optional[GeminiClient] = None
) -> str:
    """
    End-to-end RAG process:
      1. Save user message first.
      2. Build summarized context from DB for user.
      3. Build contextual prompt.
      4. Generate AI answer.
      5. Save AI answer
      6: Return formatted answer
    """
    try:
        # Initialize the generative model client
        client = gemini_client or GeminiClient()
        
        # --- Step 1: Save the message sent from a user to an agent. ---
        save_chat_message(user_id, "user", user_message, persona_id, touchpoint_id, keywords)

        # --- Step 2: 
        summarized_context = build_context_summary(user_id, touchpoint_id, user_message, client)

        # --- Step 3: Build final LLM prompt ---
        prompt = _build_prompt(user_message, summarized_context, target_language)
        logger.info(f"🧠 Final Prompt:\n{prompt}")

        # --- Step 4: Generate AI answer using Gemini ---
        answer = client.generate_content(prompt, temperature=temperature_score)
        if not answer:
            return "⚠️ I couldn't find enough information to answer that confidently."

        # --- Step 5: Save AI answer back to DB ---
        save_chat_message(user_id, "bot", answer, persona_id, touchpoint_id)

        # --- Step 6: Return formatted answer ---
        return markdown.markdown(answer) if answer_in_format == "html" else answer

    except Exception as e:
        logger.exception("❌ RAG pipeline error")
        # Graceful fallback: suggest Google search
        query = user_message.replace(" ", "+")
        return f"Try <a href='https://www.google.com/search?q={query}' target='_blank'>searching Google</a>."
