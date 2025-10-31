# leoai/rag_agent.py
import logging
import markdown
from typing import Optional, List, Union, Any
from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.rag_db_manager import ChatDBManager
from leoai.rag_context_manager import ContextManager
from leoai.rag_prompt_builder import PromptBuilder
from leoai.rag_knowledge_manager import KnowledgeRetriever
from leoai.rag_intent_analyzer import IntentAnalyzer
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
        self.intent_analyzer = IntentAnalyzer(self.client)

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
            # 0. Analyze intent of user message for keywords storage purpose
            analysis = await self.intent_analyzer.analyze(user_message)
            # 1. Save user message
            await self.db.save_chat_message(
                user_id=user_id, role="user", 
                message=user_message,
                cdp_profile_id=cdp_profile_id, 
                persona_id=persona_id,
                touchpoint_id=touchpoint_id,            
                keywords=analysis["keywords"],
                last_intent_label=analysis["last_intent_label"],
                last_intent_confidence=analysis["last_intent_confidence"],
            )

            # 2. Build summarized context
            summarized_context = await self.context.build_context_summary(
                user_id, touchpoint_id, cdp_profile_id, user_message
            )

            # Optional: Cache name in Redis
            if first_name := summarized_context.get("user_profile", {}).get("first_name"):
                REDIS_CLIENT.hset(user_id, mapping={"profile_id": "", "name": first_name})

            # 3. Build contextual prompt
            final_prompt = self.prompt_builder.build_prompt(
                user_message, summarized_context, target_language
            )
            logger.info(f"🧠 Final Prompt:\n{final_prompt}")

            # 4. Generate AI answer
            answer = self.client.generate_content(final_prompt, temperature=temperature_score)
            if not answer:
                return "⚠️ I couldn't find enough information to answer confidently."

            # 5. Save AI answer
            await self.db.save_chat_message(user_id, "bot", answer, cdp_profile_id, persona_id, touchpoint_id)

            # 6. Return
            return markdown.markdown(answer) if answer_in_format == "html" else answer

        except Exception:
            logger.exception("❌ RAG pipeline error")
            return "I'm sorry, but something went wrong. Please try again later."
