import os
import torch
import hashlib
import datetime
import logging
import markdown
import psycopg
from typing import Optional
from dotenv import load_dotenv

from leoai.ai_core import GeminiClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Device Configuration ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# --- Configuration ---
DATABASE_URL = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:password@localhost:5432/customer360")
TEMPERATURE_SCORE = float(os.getenv("TEMPERATURE_SCORE", 0.86))
VECTOR_DIMENSION = 768

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGAgent")

# --- AI Components ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = GeminiClient(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base", device=device)

# --- DB Connection ---


def get_pg_conn():
    return psycopg.connect(DATABASE_URL)

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
    vector = embedding_model.encode(
        f"{role}: {message}", normalize_embeddings=True).tolist()

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
            """, (user_id, persona_id, touchpoint_id, role, message, keywords, vector))

            logger.info(f"💾 Stored message + vector for user={user_id}")

# --- Semantic Retrieval via raw SQL ---


def retrieve_semantic_context(user_id: str, question: str, k: int = 4) -> str:
    question_vector = embedding_model.encode(
        f"user: {question}", normalize_embeddings=True).tolist()

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
            return "\n".join(messages)

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
        save_chat_message(user_id, "user", question,
                          persona_id, touchpoint_id, keywords)
        context = retrieve_semantic_context(user_id, question)

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

        logger.info(f"🧠 Sending to Gemini \n {prompt} ")
        answer = gemini_client.generate_content(prompt, temperature_score)

        if not answer:
            return "❌ No answer generated."

        save_chat_message(user_id, "bot", answer, persona_id, touchpoint_id)
        return markdown.markdown(answer) if answer_in_format == "html" else answer

    except Exception as e:
        logger.exception("❌ RAG execution error")
        fallback = question.replace(" ", "+")
        return f"Try <a href='https://www.google.com/search?q={fallback}' target='_blank'>searching Google</a>."
