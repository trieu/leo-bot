import time
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from redis import Redis
import requests

from leoai.ai_chatbot import ask_question, translate_text, detect_language, extract_data_from_chat_message_by_ai
from leoai.leo_datamodel import Message, UpdateProfileEvent, ChatMessage, TrackedEvent
from leoai.rag_agent import RAGAgent
from config import (
    HOSTNAME, LEOBOT_DEV_MODE, REDIS_HOST, REDIS_PORT, RATE_LIMIT_WINDOW_SECONDS,
    RATE_LIMIT_MAX_MESSAGES, BASE_URL_FB_MSG, FB_PAGE_ACCESS_TOKEN, FB_VERIFY_TOKEN,
    GEMINI_API_KEY, RESOURCES_DIR, TEMPLATES_DIR
)

print(f"HOSTNAME: {HOSTNAME}")
print(f"LEOBOT_DEV_MODE: {LEOBOT_DEV_MODE}")

# Redis Client to get User Session
REDIS_CLIENT = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init FAST API leobot
leobot = FastAPI()

# Initialize a global RAG agent to be reused
rag_agent = RAGAgent()

# CORS
origins = ["*"]
leobot.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
leobot.mount("/resources", StaticFiles(directory=RESOURCES_DIR),
             name="resources")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def is_safe_to_answer(visitor_id: str) -> bool:
    """ check chat_rate_limit

    Args:
        visitor_id (str): the visitor id

    Returns:
        bool: true if user not send max messages in  time window
    """
    key = f"chat_rate_limit:{visitor_id}"
    now = int(time.time() * 1000)  # current timestamp in ms
    window_start = now - (RATE_LIMIT_WINDOW_SECONDS * 1000)

    # remove entries older than the window
    REDIS_CLIENT.zremrangebyscore(key, 0, window_start)
    # count current entries in window
    current_count = REDIS_CLIENT.zcard(key)

    if current_count >= RATE_LIMIT_MAX_MESSAGES:
        return False

    # add new message timestamp
    REDIS_CLIENT.zadd(key, {str(now): now})
    # set expiration to auto-clean old data
    REDIS_CLIENT.expire(key, RATE_LIMIT_WINDOW_SECONDS)
    return True

##### API handlers #####


@leobot.get("/is-ready", response_class=JSONResponse)
@leobot.post("/is-ready", response_class=JSONResponse)
async def is_leobot_ready():
    isReady = bool(GEMINI_API_KEY and GEMINI_API_KEY.strip())
    return {"ok": isReady}


# FB Webhook verification (GET)
@leobot.get("/fb-webhook", response_class=JSONResponse)
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == FB_VERIFY_TOKEN:
        logger.info("FB Webhook verified successfully.")
        return PlainTextResponse(content=challenge)
    else:
        logger.warning("FB Webhook verification failed.")
        return JSONResponse(status_code=403, content={"error": "Verification failed"})

# FB Webhook message receiver (POST)


@leobot.post("/fb-webhook", response_class=JSONResponse)
async def webhook_handler(request: Request):
    try:
        body = await request.json()
        entries = body.get("entry", [])

        for entry in entries:
            messaging_events = entry.get("messaging", [])
            for event in messaging_events:
                sender_id = event["sender"]["id"]
                message_data = event.get("message", {})
                user_msg = message_data.get("text")

                if not user_msg:
                    continue  # skip non-text messages

                logger.info(f"FB user '{sender_id}' said: {user_msg}")

                # Generate AI reply
                persona_id = "fb_user"
                touchpoint_id = "facebook"

                # reply by AI chatbot
                ai_reply = await rag_agent.process_chat_message(
                    user_id=sender_id,
                    user_message=user_msg,
                    persona_id=persona_id,
                    cdp_profile_id='',
                    touchpoint_id=touchpoint_id
                )

                logger.info(f"AI reply to '{sender_id}': {ai_reply}")

                # Send reply
                send_message_to_facebook(sender_id, ai_reply)
    except Exception as e:
        logger.exception("Error in FB webhook handler")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"ok": True}


def send_message_to_facebook(recipient_id: str, message_text: str):
    url = f"{BASE_URL_FB_MSG}?access_token={FB_PAGE_ACCESS_TOKEN}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text},
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Sent message to {recipient_id}: {message_text}")
    except Exception as e:
        logger.error(f"Failed to send message to FB user {recipient_id}: {e}")


@leobot.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@leobot.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME,
            "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, 'timestamp': ts}
    return templates.TemplateResponse("index.html", data)


@leobot.get("/demo-chatbot-ishop", response_class=HTMLResponse)
async def demo_chat_in_ishop(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME,
            "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, 'timestamp': ts}
    return templates.TemplateResponse("demo-chatbot-ishop.html", data)


@leobot.get("/get-visitor-info", response_class=JSONResponse)
async def get_visitor_info(visitor_id: str):
    # Check GEMINI_API_KEY readiness
    if not isinstance(GEMINI_API_KEY, str) or not GEMINI_API_KEY.strip():
        return {"answer": "GEMINI_API_KEY is empty", "error_code": 501}

    # Validate visitor_id
    if not visitor_id:
        return {"answer": "visitor_id is empty", "error": True, "error_code": 500}

    # Fetch profile_id from Redis
    profile_id = REDIS_CLIENT.hget(visitor_id, "profile_id")

    # If no profile_id exists
    if not profile_id:
        # Initialize default values in Redis
        REDIS_CLIENT.hset(visitor_id, mapping={"profile_id": "", "name": "user"})
        return {"answer": "user", "error_code": 0}

    # Fetch the name (ensure it is a string)
    name = str(REDIS_CLIENT.hget(visitor_id, "name") or "user")
    return {"answer": name, "error_code": 0}


# the main API of chatbot
@leobot.post("/ask", response_class=JSONResponse)
async def web_handler(msg: Message):
    visitor_id = msg.visitor_id
    if len(visitor_id) == 0:
        return {"answer": "visitor_id is empty ", "error": True, "error_code": 500}

    # check to make sure it safe, avoid API flooding
    profile_id = REDIS_CLIENT.hget(visitor_id, 'profile_id')
    safe = is_safe_to_answer(visitor_id)
    print("profile_id " + str(profile_id) + " safe " + str(safe) + " visitor_id " + str(visitor_id))

    question = msg.question
    if len(question) > 1000:
        return {"answer": "Question must be less than 1000 characters!", "error": True, "error_code": 510}

    if safe:
        answer_in_language = msg.answer_in_language
        format = msg.answer_in_format
        persona_id = msg.persona_id
        touchpoint_id = msg.touchpoint_id

        # Generate AI reply
        answer = await rag_agent.process_chat_message(
            user_id=visitor_id,
            user_message=question,
            persona_id=persona_id,
            cdp_profile_id=profile_id,
            touchpoint_id=touchpoint_id,
            target_language=answer_in_language,
            answer_in_format=format
        )

        logger.info(f"Answer for visitor {visitor_id}: {answer}")
        data = {"question": question, "answer": answer,
                "visitor_id": visitor_id, "error_code": 0}
    else:
        logger.warning(f"Rate limit exceeded or invalid profile for visitor_id: {visitor_id}")
        error_message = "You are sending messages too quickly. Please wait a moment."
        data = {"answer": error_message, "error": True, "error_code": 429}
    return data
