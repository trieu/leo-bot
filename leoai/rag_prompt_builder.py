# leoai/rag_prompt_builder.py
import json
from datetime import datetime

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

class PromptBuilder:
    """Constructs the contextual prompt string for Gemini or any LLM."""

    def build_prompt(self, question: str, context_model: dict, target_language: str) -> str:
        user_context = context_model.get("user_context", {})
        timestamp = user_context.get("datetime")
        try:
            if timestamp:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
                ts_str = dt.strftime("%A, %B %d, %Y at %I:%M %p")
            else:
                ts_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        except Exception:
            ts_str = "Timestamp unavailable"

        user_profile_str = json.dumps(context_model.get("user_profile", {}), ensure_ascii=False, indent=2)
        user_context_str = json.dumps(user_context, ensure_ascii=False, indent=2)
        context_summary = context_model.get("context_summary", "")
        context_keywords = ", ".join(context_model.get("context_keywords", [])) or "None"

        return PROMPT_TEMPLATE.format(
            target_language=target_language,
            datetime=ts_str,
            user_profile=user_profile_str,
            user_context=user_context_str,
            context_summary=context_summary,
            context_keywords=context_keywords,
            question=question.strip()
        )
