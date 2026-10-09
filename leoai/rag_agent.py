import logging
import asyncio
import re
from html import escape
from urllib.parse import urlencode
from typing import Any, List, Optional, TypedDict, Union
from langgraph.graph import END, START, StateGraph
from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.rag_db_manager import (
    ChatDBManager, NEARBY_PLACES_LIMIT, NearbyLocationUnavailable,
)
from leoai.rag_context_manager import ContextManager
from leoai.rag_prompt_builder import AgentOrchestrator
from leoai.rag_knowledge_manager import KnowledgeRetriever
from main_config import REDIS_CLIENT

logger = logging.getLogger("RAGAgent")
logger.setLevel(logging.INFO)
DOCUMENT_CHAT_TOUCHPOINT_ID = "document_agent"


class ChatMessageState(TypedDict, total=False):
    user_id: str
    user_message: str
    cdp_profile_id: str | None
    persona_id: str | None
    touchpoint_id: str | None
    target_language: str
    answer_in_format: str
    temperature_score: float
    keywords: list[str] | None
    latitude: float | None
    longitude: float | None
    touchpoint_name: str
    touchpoint_description: str
    touchpoint_type: str
    touchpoint_keywords: list[str] | None
    result_limit: int | None
    context: str
    terms: list[str]
    matching_places: list[dict]
    summarized_context: dict
    user_context: dict
    nearby_places: list[dict]
    selected_place: dict | None
    prompt_router: Any
    response: str


def is_document_chat(context: str, persona_id: str | None) -> bool:
    return (
        context.strip().lower() == "agent"
        and (persona_id or "").strip().lower() == "personal_assistant"
    )

GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|xin chao|xin chào|chao|chào|good morning|"
    r"good afternoon|good evening)[!. ]*$",
    re.IGNORECASE,
)
PLACE_SELECTION_PATTERN = re.compile(
    r"^(?:option|choice|select|pick|place|chọn|so|số)?\s*([1-5])\s*[\].):\-]?$",
    re.IGNORECASE,
)
NEARBY_QUERY_PATTERN = re.compile(
    r"\b(?:near\s+me|nearby|close\s+to\s+me|around\s+me|"
    r"gần\s+(?:tôi|mình|đây)|xung\s+quanh)\b",
    re.IGNORECASE,
)


def is_greeting_message(message: str) -> bool:
    return bool(GREETING_PATTERN.fullmatch(message.strip()))


def selected_place_index(message: str, place_count: int) -> int | None:
    if place_count == 0:
        return None
    match = PLACE_SELECTION_PATTERN.fullmatch(message.strip())
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if index < place_count else None


def nearby_place_terms(message: str) -> list[str]:
    normalized = message.lower()
    if re.search(r"\b(?:church|churches|cathedrals?|nhà\s+thờ|nha\s+tho)\b", normalized):
        return ["church", "cathedral", "nhà thờ"]
    if re.search(r"\b(?:pagoda|temple|chùa|đền)\b", normalized):
        return ["pagoda", "temple", "chùa", "đền"]
    if re.search(r"\b(?:market|markets|chợ)\b", normalized):
        return ["market", "chợ"]
    if re.search(r"\b(?:cafe|coffee|restaurant|food|quán|ăn)\b", normalized):
        return ["cafe", "coffee", "restaurant", "food", "quán"]
    return []


def is_nearby_place_question(message: str) -> bool:
    normalized = message.lower()
    return bool(
        NEARBY_QUERY_PATTERN.search(normalized)
        and (
            nearby_place_terms(normalized)
            or re.search(r"\b(?:place|places|địa điểm|đi đâu)\b", normalized)
        )
    )


def nearby_result_limit(message: str, explicit_limit: int | None = None) -> int:
    """Resolve the requested count; an API result_limit overrides the question."""
    if explicit_limit is not None:
        limit = explicit_limit
    else:
        match = re.search(r"\btop\s+([+-]?\d+(?:\.\d+)?)\b", message, re.IGNORECASE)
        if match is None:
            match = re.search(
                r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s+(?:(?:nearest|closest)\s+)?"
                r"(?:church(?:es)?|cathedrals?|places?|markets?|pagodas?|temples?|"
                r"cafes?|restaurants?|nhà\s+thờ|nha\s+tho|địa\s+điểm|chùa|chợ)\b",
                message,
                re.IGNORECASE,
            )
        if match is not None:
            try:
                limit = int(match.group(1))
            except ValueError as exc:
                raise ValueError("Requested place count must be a positive integer.") from exc
        else:
            limit = NEARBY_PLACES_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Requested place count must be a positive integer.")
    return limit


def format_nearby_places_answer(
    places: list[dict], target_language: str, terms: list[str],
    answer_in_format: str = "text",
) -> str:
    is_vietnamese = (target_language or "").lower().startswith(("vi", "vietnam"))
    if "church" in terms:
        subject = "nhà thờ" if is_vietnamese else "churches"
    else:
        subject = "địa điểm" if is_vietnamese else "places"
    heading = (
        f"Các {subject} gần bạn:"
        if is_vietnamese
        else f"Nearby {subject}:"
    )
    if not places:
        message = (
            "Mình không tìm thấy địa điểm phù hợp trong bán kính hiện tại."
            if is_vietnamese
            else "I could not find a matching place within the current search radius."
        )
        return f"<p>{escape(message)}</p>" if answer_in_format == "html" else message
    lines = [heading, ""]
    items = []
    for index, place in enumerate(places, 1):
        distance = place.get("distance_meters")
        distance_text = f"{distance:.0f} m" if distance is not None else "distance unavailable"
        details = [place[key] for key in ("address", "description") if place.get(key)]
        lines.append(
            f"{index}. {place['name']} ({distance_text})"
            + (" - " + " - ".join(details) if details else "")
        )
        maps_query = ", ".join(
            value for value in (place["name"], place.get("address")) if value
        )
        maps_url = "https://www.google.com/maps/search/?" + urlencode(
            {"api": "1", "query": maps_query}
        )
        items.append(
            f'<li><a href="{escape(maps_url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">'
            f"<strong>{escape(place['name'])}</strong></a> "
            f"({escape(distance_text)})"
            + "".join(f"<br>{escape(detail)}" for detail in details)
            + "</li>"
        )
    if answer_in_format == "html":
        return f"<p>{escape(heading)}</p><ol>{''.join(items)}</ol>"
    return "\n".join(lines)


def format_place_picker(places: list[dict], target_language: str) -> str:
    is_vietnamese = target_language.lower().startswith(("vi", "vietnam"))
    heading = (
        "Chào bạn! Hãy chọn một địa điểm gần bạn để bắt đầu:"
        if is_vietnamese
        else "Hi! Pick one of these nearby places to start:"
    )
    lines = [heading, ""]
    for index, place in enumerate(places[:5], 1):
        distance = place.get("distance_meters")
        distance_text = f" ({distance:.0f} m)" if distance is not None else ""
        lines.append(f"{index}. {place.get('name', 'Unnamed place')}{distance_text}")
    return "\n".join(lines)


def format_place_selection_confirmation(place: dict, target_language: str) -> str:
    is_vietnamese = target_language.lower().startswith(("vi", "vietnam"))
    name = place.get("name", "địa điểm này" if is_vietnamese else "this place")
    if is_vietnamese:
        return f"Bạn đã chọn **{name}**. Bạn muốn biết điều gì về địa điểm này?"
    return f"You selected **{name}**. What would you like to know about it?"


class RAGAgent:
    def __init__(self, gemini_client: Optional[Union[GeminiClient, Any]] = None):
        self.client = gemini_client or GeminiClient()
        self.embedding_model = get_embedding_model()
        self.db = ChatDBManager(self.embedding_model)
        self.context = ContextManager(self.embedding_model, self.client, self.db)
        self.knowledge = KnowledgeRetriever(self.embedding_model)
        self.agent_orchestrator = AgentOrchestrator()

    async def create_geolocation_touchpoint(
        self,
        user_id: str,
        latitude: float,
        longitude: float,
        touchpoint_id: Optional[str] = None,
        name: str = "Web visitor",
        description: str = "",
        touchpoint_type: str = "web",
        keywords: Optional[List[str]] = None,
    ) -> dict:
        return await self.db.upsert_geolocation_touchpoint(
            user_id,
            latitude,
            longitude,
            touchpoint_id,
            name=name,
            description=description,
            touchpoint_type=touchpoint_type,
            keywords=keywords or [],
        )

    async def process_chat_message(
        self,
        user_id: str,
        user_message: str,
        cdp_profile_id: Optional[str] = None,
        persona_id: Optional[str] = 'personal_assistant',
        touchpoint_id: Optional[str] = None,
        target_language: str = "Vietnamese",
        answer_in_format: str = "text",
        temperature_score: float = 0.85,
        keywords: Optional[List[str]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        touchpoint_name: str = "Web visitor",
        touchpoint_description: str = "",
        touchpoint_type: str = "web",
        touchpoint_keywords: Optional[List[str]] = None,
        result_limit: int | None = None,
        context: str = "chatbot",
    ) -> str:
        """Process a chat message using the RAG agent.

        This method handles both document-based chats and nearby place queries,
        validates input parameters, and delegates the processing to the appropriate
        internal methods.


        Args:
            user_id (str): _description_
            user_message (str): _description_
            cdp_profile_id (Optional[str], optional): _description_. Defaults to None.
            persona_id (Optional[str], optional): _description_. Defaults to 'personal_assistant'.
            touchpoint_id (Optional[str], optional): _description_. Defaults to None.
            target_language (str, optional): _description_. Defaults to "Vietnamese".
            answer_in_format (str, optional): Retained for API compatibility;
                responses are always plain text. Defaults to "text".
            temperature_score (float, optional): _description_. Defaults to 0.85.
            keywords (Optional[List[str]], optional): _description_. Defaults to None.
            latitude (Optional[float], optional): _description_. Defaults to None.
            longitude (Optional[float], optional): _description_. Defaults to None.
            touchpoint_name (str, optional): _description_. Defaults to "Web visitor".
            touchpoint_description (str, optional): _description_. Defaults to "".
            touchpoint_type (str, optional): _description_. Defaults to "web".
            touchpoint_keywords (Optional[List[str]], optional): _description_. Defaults to None.
            result_limit (int | None, optional): _description_. Defaults to None.
            context (str, optional): _description_. Defaults to "chatbot".

        Raises:
            ValueError: _description_
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            str: Plain-text response.
        """
        initial_state: ChatMessageState = {
            "user_id": user_id,
            "user_message": user_message,
            "cdp_profile_id": cdp_profile_id,
            "persona_id": persona_id,
            "touchpoint_id": touchpoint_id,
            "target_language": target_language,
            "answer_in_format": answer_in_format,
            "temperature_score": temperature_score,
            "keywords": keywords,
            "latitude": latitude,
            "longitude": longitude,
            "touchpoint_name": touchpoint_name,
            "touchpoint_description": touchpoint_description,
            "touchpoint_type": touchpoint_type,
            "touchpoint_keywords": touchpoint_keywords,
            "result_limit": result_limit,
            "context": context,
        }
        try:
            result = await self._get_chat_graph().ainvoke(initial_state)
            return result.get("response", "")
        except Exception as e:
            logger.exception("❌ RAG pipeline error")
            return f"I'm sorry, but something went wrong: {e}"

    def _get_chat_graph(self):
        graph = getattr(self, "_chat_graph", None)
        if graph is not None:
            return graph

        workflow = StateGraph(ChatMessageState)
        workflow.add_node("document_chat", self._chat_document_node)
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("nearby_places", self._nearby_places_node)
        workflow.add_node("prepare_touchpoint", self._prepare_touchpoint_node)
        workflow.add_node("save_user_message", self._save_user_message_node)
        workflow.add_node("build_context", self._build_context_node)
        workflow.add_node("place_picker", self._place_picker_node)
        workflow.add_node("place_selection", self._place_selection_node)
        workflow.add_node("build_prompt", self._build_prompt_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("save_answer", self._save_answer_node)

        workflow.add_conditional_edges(
            START,
            self._route_chat_type,
            {"document_chat": "document_chat", "standard_chat": "validate_input"},
        )
        workflow.add_edge("document_chat", END)
        workflow.add_conditional_edges(
            "validate_input",
            self._route_after_validation,
            {"nearby_places": "nearby_places", "standard_chat": "prepare_touchpoint"},
        )
        workflow.add_edge("nearby_places", END)
        workflow.add_edge("prepare_touchpoint", "save_user_message")
        workflow.add_edge("save_user_message", "build_context")
        workflow.add_conditional_edges(
            "build_context",
            self._route_after_context,
            {
                "place_picker": "place_picker",
                "place_selection": "place_selection",
                "generate_answer": "build_prompt",
            },
        )
        workflow.add_edge("place_picker", END)
        workflow.add_edge("place_selection", END)
        workflow.add_edge("build_prompt", "generate_answer")
        workflow.add_edge("generate_answer", "save_answer")
        workflow.add_edge("save_answer", END)

        self._chat_graph = workflow.compile()
        return self._chat_graph

    @staticmethod
    def _route_chat_type(state: ChatMessageState) -> str:
        if is_document_chat(state["context"], state.get("persona_id")):
            return "document_chat"
        return "standard_chat"

    def _validate_input_node(self, state: ChatMessageState) -> dict:
        if (state.get("latitude") is None) != (state.get("longitude") is None):
            raise ValueError("latitude and longitude must be provided together")
        if not is_nearby_place_question(state["user_message"]):
            return {}
        return {
            "terms": nearby_place_terms(state["user_message"]),
            "result_limit": nearby_result_limit(
                state["user_message"], state.get("result_limit")
            ),
        }

    @staticmethod
    def _route_after_validation(state: ChatMessageState) -> str:
        if is_nearby_place_question(state["user_message"]):
            return "nearby_places"
        return "standard_chat"

    async def _chat_document_node(self, state: ChatMessageState) -> dict:
        response = await self._process_document_chat(
            state["user_id"],
            state["user_message"],
            state.get("cdp_profile_id"),
            state.get("persona_id"),
            state["target_language"],
            state["answer_in_format"],
            state["temperature_score"],
        )
        return {"response": response}

    async def _nearby_places_node(self, state: ChatMessageState) -> dict:
        try:
            places = await self.db.find_nearby_places(
                state.get("touchpoint_id"),
                state["terms"],
                state["result_limit"],
                user_id=state["user_id"],
                latitude=state.get("latitude"),
                longitude=state.get("longitude"),
            )
        except NearbyLocationUnavailable:
            logger.info(
                "Nearby search requires location for visitor %s",
                state["user_id"],
            )
            response = (
                "Hãy chia sẻ vị trí của bạn để mình tìm các địa điểm gần nhất."
                if (state["target_language"] or "").lower().startswith(("vi", "vietnam"))
                else "Please share your location so I can find nearby places."
            )
            return {"response": response}

        response = format_nearby_places_answer(
            places, state["target_language"], state["terms"]
        )
        for role, message in (("user", state["user_message"]), ("bot", response)):
            await self.db.save_chat_message(
                state["user_id"],
                role,
                message,
                state.get("cdp_profile_id"),
                state.get("persona_id"),
                state.get("touchpoint_id") or "web_leobot",
                embed=False,
            )
        return {"response": response, "matching_places": places}

    async def _prepare_touchpoint_node(self, state: ChatMessageState) -> dict:
        touchpoint_id = state.get("touchpoint_id")
        if state.get("latitude") is not None and state.get("longitude") is not None:
            touchpoint = await self.create_geolocation_touchpoint(
                state["user_id"],
                state["latitude"],
                state["longitude"],
                touchpoint_id,
                name=state["touchpoint_name"],
                description=state["touchpoint_description"],
                touchpoint_type=state["touchpoint_type"],
                keywords=state.get("touchpoint_keywords") or [],
            )
            touchpoint_id = touchpoint["touchpoint_id"]
        return {"touchpoint_id": touchpoint_id or "web_leobot"}

    async def _save_user_message_node(self, state: ChatMessageState) -> dict:
        try:
            logger.info("🧠 insert user message: %s", state["user_message"])
            await self.db.save_chat_message(
                user_id=state["user_id"],
                role="user",
                message=state["user_message"],
                cdp_profile_id=state.get("cdp_profile_id"),
                persona_id=state.get("persona_id"),
                touchpoint_id=state["touchpoint_id"],
                keywords=state.get("keywords"),
            )
        except Exception as exc:
            logger.error("❌ Failed to save user message to DB: %s", exc)
        return {}

    async def _build_context_node(self, state: ChatMessageState) -> dict:
        summarized_context = await self.context.build_context_summary(
            state["user_id"],
            state["touchpoint_id"],
            state.get("cdp_profile_id"),
            state["user_message"],
        )
        user_context = summarized_context.get("user_context", {})
        return {
            "summarized_context": summarized_context,
            "user_context": user_context,
            "nearby_places": user_context.get("nearby_places", [])[:5],
            "selected_place": user_context.get("selected_place"),
        }

    def _route_after_context(self, state: ChatMessageState) -> str:
        nearby_places = state.get("nearby_places", [])
        if (
            is_greeting_message(state["user_message"])
            and nearby_places
            and not state.get("selected_place")
        ):
            return "place_picker"
        place_choices = state["user_context"].get("place_choices", nearby_places)
        if selected_place_index(state["user_message"], len(place_choices)) is not None:
            return "place_selection"
        return "generate_answer"

    async def _place_picker_node(self, state: ChatMessageState) -> dict:
        user_context = dict(state["user_context"])
        user_context["place_choices"] = [
            dict(place) for place in state["nearby_places"]
        ]
        summary = dict(state["summarized_context"])
        summary["user_context"] = user_context
        persisted_context = dict(summary)
        persisted_context.pop("updated_at", None)
        if not await self.db.save_context_summary(
            state["user_id"],
            state["touchpoint_id"],
            state.get("cdp_profile_id"),
            persisted_context,
        ):
            raise RuntimeError("Failed to save the nearby-place choices.")
        response = format_place_picker(
            state["nearby_places"], state["target_language"]
        )
        await self.db.save_chat_message(
            state["user_id"],
            "bot",
            response,
            state.get("cdp_profile_id"),
            state.get("persona_id"),
            state["touchpoint_id"],
        )
        return {"response": response}

    async def _place_selection_node(self, state: ChatMessageState) -> dict:
        nearby_places = state.get("nearby_places", [])
        user_context = dict(state["user_context"])
        place_choices = user_context.get("place_choices", nearby_places)
        selection_index = selected_place_index(
            state["user_message"], len(place_choices)
        )
        if selection_index is None:
            raise RuntimeError("Place selection route lost its selected place.")
        selected_place = dict(place_choices[selection_index])
        user_context["selected_place"] = selected_place

        summary = dict(state["summarized_context"])
        summary["user_context"] = user_context
        persisted_context = dict(summary)
        persisted_context.pop("updated_at", None)
        if not await self.db.save_context_summary(
            state["user_id"],
            state["touchpoint_id"],
            state.get("cdp_profile_id"),
            persisted_context,
        ):
            raise RuntimeError("Failed to save the selected place.")
        response = format_place_selection_confirmation(
            selected_place, state["target_language"]
        )
        await self.db.save_chat_message(
            state["user_id"],
            "bot",
            response,
            state.get("cdp_profile_id"),
            state.get("persona_id"),
            state["touchpoint_id"],
        )
        return {"response": response, "selected_place": selected_place}

    async def _build_prompt_node(self, state: ChatMessageState) -> dict:
        summarized_context = dict(state["summarized_context"])
        user_context = dict(summarized_context.get("user_context") or {})
        selected_place = user_context.get("selected_place")
        if selected_place:
            place_knowledge = await self._retrieve_selected_place_knowledge(
                selected_place, state["user_message"]
            )
            if place_knowledge:
                user_context["selected_place_knowledge"] = place_knowledge
                summarized_context["user_context"] = user_context

        first_name = (summarized_context.get("user_profile") or {}).get("first_name")
        if first_name:
            REDIS_CLIENT.hset(
                state["user_id"], mapping={"profile_id": "", "name": first_name}
            )
        prompt_router = self.agent_orchestrator.build_prompt(
            state["user_message"],
            summarized_context,
            state["target_language"],
            state.get("persona_id"),
        )
        logger.info("🧠 Detected purpose: %s", prompt_router.purpose)
        logger.info("📝 Prompt snippet: %s...", prompt_router.prompt_text[:200])
        return {
            "summarized_context": summarized_context,
            "user_context": user_context,
            "prompt_router": prompt_router,
        }

    async def _generate_answer_node(self, state: ChatMessageState) -> dict:
        response = await self._safe_generate(
            state["prompt_router"], state["temperature_score"]
        )
        return {"response": response}

    async def _save_answer_node(self, state: ChatMessageState) -> dict:
        if state["prompt_router"].purpose == "generate_text":
            logger.info("🧠 insert bot message: %s", state["response"])
            await self.db.save_chat_message(
                state["user_id"],
                "bot",
                state["response"],
                state.get("cdp_profile_id"),
                state.get("persona_id"),
                state["touchpoint_id"],
            )
        return {}

    async def _retrieve_selected_place_knowledge(
        self, selected_place: dict, user_message: str
    ) -> str:
        retriever = getattr(self, "knowledge", None)
        retrieve = getattr(retriever, "retrieve_selected_place", None)
        if retrieve is None:
            return ""
        return await retrieve(selected_place, user_message, limit=3)

    async def _process_document_chat(
        self,
        user_id: str,
        user_message: str,
        cdp_profile_id: str | None,
        persona_id: str | None,
        target_language: str,
        answer_in_format: str,
        temperature_score: float,
    ) -> str:
        """Keep document Q&A separate from the visitor's geolocation conversation."""
        await self.db.save_chat_message(
            user_id, "user", user_message, cdp_profile_id, persona_id,
            DOCUMENT_CHAT_TOUCHPOINT_ID,
        )
        summary = await self.context.build_context_summary(
            user_id, DOCUMENT_CHAT_TOUCHPOINT_ID, cdp_profile_id, user_message,
            include_location=False,
        )
        document_context = ""
        if not is_greeting_message(user_message):
            document_context = await self.knowledge.retrieve(
                user_message, "default", user_id=user_id,
            )
        prompt = self.agent_orchestrator.build_document_prompt(
            user_message, summary, document_context, target_language,
        )
        answer = await self._safe_generate(prompt, temperature_score)
        await self.db.save_chat_message(
            user_id, "bot", answer, cdp_profile_id, persona_id,
            DOCUMENT_CHAT_TOUCHPOINT_ID,
        )
        return answer

    async def _safe_generate(self, prompt_router, temperature_score: float) -> str:
        """
        Safely execute GeminiClient methods in async context.
        Supports both sync and async method types.
        """
        # Select correct generation method
        if prompt_router.purpose == "generate_report":
            method = self.client.generate_report
        else:
            method = self.client.generate_content

        generation_kwargs: dict[str, Any] = {"temperature": temperature_score}
        if prompt_router.purpose != "generate_report" and isinstance(self.client, GeminiClient):
            generation_kwargs["system_instruction"] = prompt_router.system_instruction

        # Handle async vs sync automatically
        if asyncio.iscoroutinefunction(method):
            return await method(prompt_router.prompt_text, **generation_kwargs)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: method(prompt_router.prompt_text, **generation_kwargs)
            )
