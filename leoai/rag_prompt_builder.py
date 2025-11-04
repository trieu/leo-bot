# leoai/rag_prompt_builder.py
import json
from datetime import datetime
from typing import Dict, Optional
import logging


logger = logging.getLogger("PromptBuilder")


PROMPT_TEMPLATE = """
You are **LEO**, a truthful, forward-thinking AI assistant designed to provide accurate, contextual, and human-like responses.
You must always respond **in {target_language}**, following the same tone and style as the user unless instructed otherwise.

---

### 🧭 Core Directives

1. **Be truthful.** If info is missing or uncertain, say so clearly.
2. **Use context.** Adapt to the user profile and past conversation, never invent facts.
3. **Stay concise.** Write complete, natural sentences — no filler.
4. **Sound human.** Speak warmly and intelligently, not robotic.
5. **No hallucination.** Never make up data, names, or sources.

---

### Current Date and Time
{datetime}

### User Profile Summary
{user_profile}

### Conversation Context Summary
{user_context}

### Key Conversation Summary
{context_summary}

### Key Conversation Keywords
{context_keywords}

---

### User’s Current Question
{question}

---

### Expected Behavior

- Give clear, relevant, and truthful answers using all context.
- Ask for clarification if the question is vague.
- Return full, working code when coding is requested.
- Explain concepts with short examples or analogies.
- End with an insightful remark or takeaway.

"""


class PromptRouter:
    """Holds the built prompt and inferred purpose of the request."""
    def __init__(self, prompt_text: str, purpose: str):
        self.prompt_text = prompt_text
        self.purpose = purpose

    def __repr__(self):
        return f"PromptRouter(purpose={self.purpose!r}, prompt_length={len(self.prompt_text)})"


class AgentOrchestrator:
    """Constructs contextual prompt strings and detects intent for routing."""

    def build_prompt(self, question: str, context_model: Dict, target_language: str) -> PromptRouter:
        user_context = context_model.get("user_context", {})
        timestamp = user_context.get("datetime")

        # Format timestamp
        ts_str = self._format_timestamp(timestamp)

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
        purpose = self.detect_purpose(question)

        return PromptRouter(prompt_text=prompt_text, purpose=purpose)

    def detect_purpose(self, question: str, context_summary: Optional[str] = None) -> str:
        """
        Infer the user's intent based on keywords, structure, and contextual hints.
        This lightweight heuristic can later be upgraded with embedding similarity.
        """
        q = question.lower().strip()
        ctx = (context_summary or "").lower()

        # --- Primary keyword groups ---
        purpose_keywords = {
            "generate_report": [
                "report", "summary", "dashboard", "insight", "trend",
                "analytics", "statistics", "data analysis", "visualization"
            ],
            "generate_text": [
                "write", "story", "poem", "email", "post", "message",
                "draft", "explain", "summarize", "describe"
            ],
            "generate_plan": [
                "plan", "strategy", "outline", "schedule",
                "proposal", "roadmap", "timeline"
            ],
            "generate_code": [
                "code", "script", "function", "query", "api", "algorithm"
            ],
            "generate_answer": [
                "what", "how", "why", "can i", "is it", "should i"
            ]
        }

        # --- Scoring mechanism ---
        scores = {purpose: 0 for purpose in purpose_keywords}

        for purpose, keywords in purpose_keywords.items():
            for kw in keywords:
                if kw in q:
                    scores[purpose] += 2  # direct keyword boost
                if kw in ctx:
                    scores[purpose] += 1  # context hint boost

        # --- Structural hints ---
        if q.endswith("?"):
            scores["generate_answer"] += 1.5

        if "data" in q and ("show" in q or "graph" in q):
            scores["generate_report"] += 2

        # --- Choose best-scoring purpose ---
        best_purpose = max(scores, key=scores.get)
        confidence = scores[best_purpose]

        # --- Confidence threshold logic ---
        if confidence < 2:
            best_purpose = "generate_text"  # default fallback

        logger.debug(f"[detect_purpose] Question='{question}' → {best_purpose} (scores={scores})")
        return best_purpose

    @staticmethod
    def _format_timestamp(timestamp: Optional[str]) -> str:
        """Helper to format datetime safely."""
        if not timestamp:
            return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            return dt.strftime("%A, %B %d, %Y at %I:%M %p")
        except Exception:
            return "Timestamp unavailable"