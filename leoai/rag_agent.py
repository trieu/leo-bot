import os

import hashlib
import datetime
import logging
import markdown

from typing import Optional
from dotenv import load_dotenv

from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.db_utils import get_pg_conn

load_dotenv()


# --- Configuration ---

TEMPERATURE_SCORE = float(os.getenv("TEMPERATURE_SCORE", 0.86))
VECTOR_DIMENSION = 768

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGAgent")

# --- AI Components ---
gemini_client = GeminiClient()
embedding_model = get_embedding_model()

# --- SHA256 hashing for duplicate check ---


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

# --- Save message + embedding ---


def save_chat_message(
    user_id: str,
    role: str,
    message: str,
    persona_id: Optional[str] = None,
    touchpoint_id: Optional[str] = None,
    keywords: Optional[list[str]] = None
):
    message_hash = sha256_hash(message)        
    message_vector = embedding_model.encode(f"{role}: {message}", normalize_embeddings=True).tolist()

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM chat_messages WHERE message_hash = %s AND user_id = %s
            """, (message_hash, user_id))
            if cur.fetchone():
                logger.info(
                    f"⚠️ Duplicate message from user={user_id}, role={role}")
                return

            cur.execute("""
                INSERT INTO chat_messages
                (user_id, persona_id, touchpoint_id, role, message, message_hash, keywords, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            """, (user_id, persona_id, touchpoint_id, role, message, message_hash, keywords))

            cur.execute("""
                INSERT INTO chat_history_embeddings
                (user_id, persona_id, touchpoint_id, role, message, keywords, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            """, (user_id, persona_id, touchpoint_id, role, message, keywords, message_vector))

            logger.info(f"💾 Stored message + vector for user={user_id}")

# --- Semantic Retrieval via raw SQL ---


def retrieve_semantic_context(user_id: str, question: str, k: int = 20, max_length: int = 5000) -> str:    
    question_vector = embedding_model.encode(f"user: {question}", normalize_embeddings=True).tolist()

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT message
                FROM chat_history_embeddings
                WHERE user_id = %s
                ORDER BY embedding <#> (%s)::vector
                LIMIT %s
            """, (user_id, question_vector, k))

            rows = cur.fetchall()
            messages = [row[0] for row in rows]
            logger.info(f"🔍 Retrieved {len(messages)} history messages")
            # --- ADD THIS PART ---
            full_context = "\n".join(messages)
            if len(full_context) > max_length:
                logger.warning(f"⚠️ Context length ({len(full_context)}) exceeds max_length ({max_length}). Truncating.")
                return full_context[:max_length]
            # --- END ---
            
            return full_context


# --- Context Summarization ---

def summarize_context(context: str, target_language: str = "Vietnamese", 
                      max_context_length: int = 2000, temperature_score: float = 0.2) -> str:
    """Uses the AI to summarize a long context."""
    if not context:
        return ""
        
    try:
        # Check if the context is already short enough
        if len(context) < max_context_length:
            return context

        logger.info(f"📝 Context is long ({len(context)} chars). Summarizing...")
        
        summary_prompt = f"""
        <s>[INST] Summarize the following conversation history concisely in {target_language}.
        Focus on the key topics, user needs, and important details mentioned.

        [Conversation History]
        {context}
        [/INST]</s>
        """
        
        summary = gemini_client.generate_content(summary_prompt, temperature_score=temperature_score) # Use low temp for factual summary
        logger.info("✅ Context summarized.")
        return summary if summary else context # Fallback to original context if summary fails
        
    except Exception as e:
        logger.error(f"❌ Error during context summarization: {e}")
        return context # Return original context on error

# --- Main RAG Agent Entry Point ---


def ask_question_rag(
    user_id: str,
    question: str,
    persona_id: Optional[str] = None,
    touchpoint_id: Optional[str] = None,
    target_language: str = "Vietnamese",
    answer_in_format: str = "text",
    temperature_score: float = TEMPERATURE_SCORE,
    keywords: Optional[list[str]] = None
) -> str:
    now = datetime.datetime.now().strftime("%c")

    try:
        # 1. save chat messages into database
        save_chat_message(user_id, "user", question, persona_id, touchpoint_id, keywords)
        
        # 2. Retrieve the context from the database
        retrieved_context = retrieve_semantic_context(user_id, question)
        
        # 3. Check the length and summarize ONLY if it's too long
        context = summarize_context(retrieved_context, target_language, max_context_length=900)
        logger.info(f"📝 summarize_context returned length ({len(context)} chars)")
    
        # 4. Construct the final prompt
        prompt = f"""
<s> [INST] You are LEO, a helpful AI assistant.
- DateTime: {now}
- Answer in: {target_language}

[User History]
{context}

[User Question]
{question}
[/INST] </s>
""".strip()

        # 5. Send prompt to LLM AI model
        logger.info(f"🧠 Sending to Gemini \n {prompt} ")
        answer = gemini_client.generate_content(prompt, temperature_score)

        if not answer:
            return "❌ No answer generated."

        # 6. save the answer of AI model
        save_chat_message(user_id, "bot", answer, persona_id, touchpoint_id)
        
        # 7. Return the answer
        return markdown.markdown(answer) if answer_in_format == "html" else answer

    except Exception as e:
        logger.exception("❌ RAG execution error")
        fallback = question.replace(" ", "+")
        return f"Try <a href='https://www.google.com/search?q={fallback}' target='_blank'>searching Google</a>."
