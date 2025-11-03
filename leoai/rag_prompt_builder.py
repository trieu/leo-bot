# leoai/rag_prompt_builder.py
import json
from datetime import datetime
from typing import Optional


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


class PromptRouter:
    """Holds the built prompt and inferred purpose of the request."""
    def __init__(self, prompt_text: str, purpose: str):
        self.prompt_text = prompt_text
        self.purpose = purpose

    def __repr__(self):
        return f"PromptRouter(purpose={self.purpose!r}, prompt_length={len(self.prompt_text)})"


class PromptBuilder:
    """Constructs contextual prompt strings and detects intent for routing."""

    def build_prompt(self, question: str, context_model: dict, target_language: str) -> PromptRouter:
        user_context = context_model.get("user_context", {})
        timestamp = user_context.get("datetime")

        # Format timestamp
        try:
            if timestamp:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
                ts_str = dt.strftime("%A, %B %d, %Y at %I:%M %p")
            else:
                ts_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        except Exception:
            ts_str = "Timestamp unavailable"

        # Prepare context
        user_profile_str = json.dumps(context_model.get("user_profile", {}), ensure_ascii=False, indent=2)
        user_context_str = json.dumps(user_context, ensure_ascii=False, indent=2)
        context_summary = context_model.get("context_summary", "")
        context_keywords = ", ".join(context_model.get("context_keywords", [])) or "None"

        # Build the final formatted prompt
        prompt_text = PROMPT_TEMPLATE.format(
            target_language=target_language,
            datetime=ts_str,
            user_profile=user_profile_str,
            user_context=user_context_str,
            context_summary=context_summary,
            context_keywords=context_keywords,
            question=question.strip()
        )

        # Detect purpose
        purpose = self.detect_purpose(question, context_summary)

        return PromptRouter(prompt_text=prompt_text, purpose=purpose)

    def detect_purpose(self, question: str, context_summary: Optional[str] = None) -> str:
        """Heuristically infer the user's intent (purpose)."""
        q = question.lower()

        # quick keyword-based heuristic
        report_words = ["report", "summary", "dashboard", "insight", "trend", "data analysis"]
        creative_words = ["write", "story", "poem", "email", "post", "draft"]
        planning_words = ["plan", "strategy", "outline", "schedule", "proposal"]

        if any(word in q for word in report_words):
            return "generate_report"
        if any(word in q for word in creative_words):
            return "generate_text"
        if any(word in q for word in planning_words):
            return "generate_plan"

        # fallback logic based on context
        if context_summary and "analytics" in context_summary.lower():
            return "generate_report"

        # default purpose
        return "generate_text"
