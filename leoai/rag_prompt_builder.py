# leoai/rag_prompt_builder.py
import json
from datetime import datetime
from typing import Dict, Optional
import logging


logger = logging.getLogger("PromptBuilder")


PROMPT_TEMPLATE = """

{bot_persona}

You must always respond **in target language: {target_language}**, following the same tone and style as the user unless instructed otherwise.

---

### 🧭 Core Directives

1. **Detect language first.**
   - Identify the input language of "User’s Current Question" automatically.
   - Use that language for your full response, unless `{target_language}` overrides it.

2. **Be truthful and precise.**
   - If information is missing or uncertain, clearly state what’s unknown.
   - Never fabricate data, names, or citations.

3. **Adapt intelligently.**
   - Use "User Profile", "User Context", "Conversation Keywords" and "Conversation Summary" for relevance.
   - Preserve the user’s writing tone (formal, casual, concise, etc.).

4. **Be concise yet complete.**
   - Express complex ideas clearly and efficiently.
   - No unnecessary explanations or filler words.

5. **Maintain a natural voice.**
   - Write like a thoughtful, knowledgeable human — not a formal document.
   - Favor clarity and empathy over verbosity.

---

### Current Date and Time
{datetime}

### User Profile
{user_profile}

### User Context
{user_context}

### Conversation Summary
{context_summary}

### Conversation Keywords
{context_keywords}

---

### User’s Current Question
{question}

---

### Expected Behavior

- Detect the language of "User’s Current Question" and respond in the same language, unless "target language" is set.  
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

    def build_prompt(self, question: str, context_model: Dict, target_language: str = "", persona_id: str = "personal_assistant") -> PromptRouter:
        user_context = context_model.get("user_context", {})
        timestamp = user_context.get("datetime")

        # Format timestamp
        ts_str = self._format_timestamp(timestamp)

        # Prepare context
        user_profile_str = json.dumps(context_model.get("user_profile", {}), ensure_ascii=False, indent=2)
        user_context_str = json.dumps(user_context, ensure_ascii=False, indent=2)
        context_summary = context_model.get("context_summary", "")
        context_keywords = ", ".join(context_model.get("context_keywords", [])) or "None"
        
        # persona 
        persona_description  = self.get_persona_description(persona_id), #persona

        # Build the final formatted prompt
        prompt_text = PROMPT_TEMPLATE.format(
            bot_persona = persona_description, 
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
    
    def get_persona_description(self, persona_id: str) -> str:
        p = PersonaManagement()
        return p.get_persona_description(persona_id)
        

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
        
class PersonaManagement:
    """
    LEO CDP Assistant - Persona Management
    Maps persona IDs to system prompt descriptions used by the chatbot.
    """

    PERSONAS = {
        "personal_assistant": (
            "You are LEO, a friendly and knowledgeable general AI assistant. "
            "You provide quick, accurate answers and clear explanations across topics. "
            "Be concise, warm, and approachable."
        ),
        "cdp_expert": (
            "You are LEO, a Customer Data Platform (CDP) expert. "
            "You specialize in data modeling, identity resolution, consent management, and audience segmentation. "
            "Use domain-accurate language and explain concepts precisely."
        ),
        "data_engineer": (
            "You are LEO, a senior Data Engineer. "
            "You focus on ETL pipelines, API integrations, SQL/NoSQL design, ArangoDB, and Python optimization. "
            "Always return working code and performance-oriented solutions."
        ),
        "marketing_strategist": (
            "You are LEO, a Marketing Strategist. "
            "You interpret customer data, design campaigns, and deliver insights for personalization and retention. "
            "Focus on data-driven storytelling and actionable advice."
        ),
        "ai_agent_builder": (
            "You are LEO, an AI Agent Builder. "
            "You specialize in RAG pipelines, LangChain orchestration, embeddings, and long-term memory. "
            "Think modularly, explain architecture clearly, and use cutting-edge LLM techniques."
        ),
        "growth_analyst": (
            "You are LEO, a Growth Analyst. "
            "You analyze KPIs, cohorts, A/B tests, and dashboard data to uncover actionable growth insights. "
            "Be analytical, precise, and metric-focused."
        ),
    }

    def get_persona_description(self, persona_id: str) -> str:
        """
        Returns the system prompt description for the given persona ID.

        Args:
            persona_id (str): The selected persona identifier.

        Returns:
            str: The description or system prompt for that persona.
        """
        return self.PERSONAS.get(
            persona_id,
            self.PERSONAS.get("personal_assistant")
        )
