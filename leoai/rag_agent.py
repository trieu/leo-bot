import logging
import markdown
import asyncio
import re
from typing import Optional, List, Union, Any
from leoai.ai_core import GeminiClient, get_embedding_model
from leoai.rag_db_manager import ChatDBManager, NEARBY_PLACES_LIMIT
from leoai.rag_context_manager import ContextManager
from leoai.rag_prompt_builder import AgentOrchestrator
from leoai.rag_knowledge_manager import KnowledgeRetriever
from main_config import REDIS_CLIENT

logger = logging.getLogger("RAGAgent")
logger.setLevel(logging.INFO)

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
    if re.search(r"\b(?:church|churches|cathedral|nhà thờ|nha tho)\b", normalized):
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


def format_nearby_places_answer(
    places: list[dict], target_language: str, terms: list[str]
) -> str:
    is_vietnamese = target_language.lower().startswith(("vi", "vietnam"))
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
        return (
            "Mình không tìm thấy địa điểm phù hợp trong bán kính hiện tại."
            if is_vietnamese
            else "I could not find a matching place within the current search radius."
        )
    lines = [heading]
    for index, place in enumerate(places, 1):
        distance = place.get("distance_meters")
        distance_text = f"{distance:.0f} m" if distance is not None else "distance unavailable"
        details = place.get("address") or place.get("description") or ""
        lines.append(f"{index}. {place['name']} ({distance_text}) - {details}")
    return "\n".join(lines)


def format_place_picker(places: list[dict], target_language: str) -> str:
    is_vietnamese = target_language.lower().startswith(("vi", "vietnam"))
    heading = (
        "Chào bạn! Hãy chọn một địa điểm gần bạn để bắt đầu:"
        if is_vietnamese
        else "Hi! Pick one of these nearby places to start:"
    )
    lines = [heading]
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
    ) -> str:
        try:
            if (latitude is None) != (longitude is None):
                raise ValueError("latitude and longitude must be provided together")
            if latitude is not None and longitude is not None:
                touchpoint = await self.create_geolocation_touchpoint(
                    user_id,
                    latitude,
                    longitude,
                    touchpoint_id,
                    name=touchpoint_name,
                    description=touchpoint_description,
                    touchpoint_type=touchpoint_type,
                    keywords=touchpoint_keywords or [],
                )
                touchpoint_id = touchpoint["touchpoint_id"]
            touchpoint_id = touchpoint_id or "web_leobot"
            try:
                # 1️⃣ Save user message (sync)
                logger.info(f"🧠 insert user message: {user_message}")
                await self.db.save_chat_message(
                    user_id=user_id,
                    role="user",
                    message=user_message,
                    cdp_profile_id=cdp_profile_id,
                    persona_id=persona_id,
                    touchpoint_id=touchpoint_id,
                    keywords=keywords,
                )
            except Exception as e:
                logger.error(f"❌ Failed to save user message to DB: {e}")

            # 2️⃣ Build summarized context (async)
            summarized_context = await self.context.build_context_summary(
                user_id, touchpoint_id, cdp_profile_id, user_message
            )

            user_context = summarized_context.get("user_context", {})
            nearby_places = user_context.get("nearby_places", [])[:5]
            selected_place = user_context.get("selected_place")

            if is_nearby_place_question(user_message) and touchpoint_id:
                terms = nearby_place_terms(user_message)
                matching_places = await self.db.find_nearby_places(
                    touchpoint_id, terms, NEARBY_PLACES_LIMIT
                )
                final_answer = format_nearby_places_answer(
                    matching_places, target_language, terms
                )
                await self.db.save_chat_message(
                    user_id,
                    "bot",
                    final_answer,
                    cdp_profile_id,
                    persona_id,
                    touchpoint_id,
                )
                return (
                    markdown.markdown(final_answer)
                    if answer_in_format == "html"
                    else final_answer
                )

            if is_greeting_message(user_message) and nearby_places and not selected_place:
                final_answer = format_place_picker(nearby_places, target_language)
                await self.db.save_chat_message(
                    user_id,
                    "bot",
                    final_answer,
                    cdp_profile_id,
                    persona_id,
                    touchpoint_id,
                )
                return (
                    markdown.markdown(final_answer)
                    if answer_in_format == "html"
                    else final_answer
                )

            selection_index = selected_place_index(user_message, len(nearby_places))
            if selection_index is not None and not selected_place:
                selected_place = nearby_places[selection_index]
                user_context["selected_place"] = selected_place
                summarized_context["user_context"] = user_context
                persisted_context = dict(summarized_context)
                persisted_context.pop("updated_at", None)
                await self.db.save_context_summary(
                    user_id, touchpoint_id, cdp_profile_id, persisted_context
                )
                final_answer = format_place_selection_confirmation(
                    selected_place, target_language
                )
                await self.db.save_chat_message(
                    user_id,
                    "bot",
                    final_answer,
                    cdp_profile_id,
                    persona_id,
                    touchpoint_id,
                )
                return (
                    markdown.markdown(final_answer)
                    if answer_in_format == "html"
                    else final_answer
                )

            # 3️⃣ Cache user info in Redis
            user_profile = summarized_context.get("user_profile", {})
            first_name = user_profile.get("first_name")
            if first_name:
                REDIS_CLIENT.hset(user_id, mapping={"profile_id": "", "name": first_name})

            # 4️⃣ Build contextual prompt
            prompt_router = self.agent_orchestrator.build_prompt(
                user_message, summarized_context, target_language, persona_id
            )

            logger.info(f"🧠 Detected purpose: {prompt_router.purpose}")
            logger.info(f"📝 Prompt snippet: {prompt_router.prompt_text[:200]}...")

            # 5️⃣ Generate AI answer (runs sync client safely in thread)
            final_answer = await self._safe_generate(prompt_router, temperature_score)

            # 6️⃣ Save AI response
            if prompt_router.purpose == "generate_text":
                logger.info(f"🧠 insert bot message: {final_answer}")
                await self.db.save_chat_message(
                    user_id, "bot", final_answer, cdp_profile_id, persona_id, touchpoint_id
                )

            # 7️⃣ Return formatted answer
            if answer_in_format == "html":
                return markdown.markdown(final_answer)
            return final_answer

        except Exception as e:
            logger.exception("❌ RAG pipeline error")
            return f"I'm sorry, but something went wrong: {e}"

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

        # Handle async vs sync automatically
        if asyncio.iscoroutinefunction(method):
            return await method(prompt_router.prompt_text, temperature=temperature_score)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: method(prompt_router.prompt_text, temperature=temperature_score)
            )
