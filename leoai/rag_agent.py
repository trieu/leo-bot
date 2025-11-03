# leoai/rag_agent.py
import logging
import markdown
from typing import Optional, List, Union, Any
from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.rag_db_manager import ChatDBManager
from leoai.rag_context_manager import ContextManager
from leoai.rag_prompt_builder import PromptBuilder
from leoai.rag_knowledge_manager import KnowledgeRetriever
from main_config import REDIS_CLIENT


logger = logging.getLogger("RAGAgent")

class RAGAgent:
    def __init__(self, gemini_client: Optional[Union[GeminiClient, Any]] = None):
        self.client = gemini_client or GeminiClient()
        self.embedding_model = get_embedding_model()
        self.db = ChatDBManager(self.embedding_model)
        self.context = ContextManager(self.embedding_model, self.client, self.db)
        self.knowledge = KnowledgeRetriever(self.embedding_model)
        self.prompt_builder = PromptBuilder()

    async def process_chat_message(
        self,
        user_id: str,
        user_message: str,
        cdp_profile_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        touchpoint_id: Optional[str] = None,
        target_language: str = "Vietnamese",
        answer_in_format: str = "text",
        temperature_score: float = 0.85,
        keywords: Optional[List[str]] = None,
    ) -> str:
        try:
            # 1. Save user message
            await self.db.save_chat_message(
                user_id=user_id, role="user", message=user_message,
                cdp_profile_id=cdp_profile_id, persona_id=persona_id,
                touchpoint_id=touchpoint_id, keywords=keywords
            )

            # 2. Build summarized context
            summarized_context = await self.context.build_context_summary(
                user_id, touchpoint_id, cdp_profile_id, user_message
            )

            # Optional: Cache name in Redis
            if first_name := summarized_context.get("user_profile", {}).get("first_name"):
                REDIS_CLIENT.hset(user_id, mapping={"profile_id": "", "name": first_name})

            # 3. Build contextual prompt
            prompt_router = self.prompt_builder.build_prompt(
                user_message, summarized_context, target_language
            )
            logger.info(f"🧠 Final Prompt:\n{prompt_router}")

            # 4. Generate AI answer
            final_answer = ''
            if prompt_router.purpose == 'generate_text':
                final_answer = self.client.generate_content(prompt_router.prompt_text, temperature=temperature_score)
                if not final_answer:
                    return "⚠️ I couldn't find enough information to answer confidently."

                # 5. Save AI answer if text only
                if answer_in_format == 'text':
                    await self.db.save_chat_message(user_id, "bot", final_answer, cdp_profile_id, persona_id, touchpoint_id)
            elif prompt_router.purpose == 'generate_report':
                answer_in_format = 'html'
                final_answer = self.client.generate_report(prompt_router.prompt_text, temperature=temperature_score)

            # 6. Return
            return markdown.markdown(final_answer) if answer_in_format == "html" else final_answer

        except Exception:
            logger.exception("❌ RAG pipeline error")
            return "I'm sorry, but something went wrong. Please try again later."
