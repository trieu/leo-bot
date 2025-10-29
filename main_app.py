
import time
import json
import logging

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests

from leoai.email_sender import EmailSender
from leoai.leo_datamodel import Message
from leoai.rag_agent import RAGAgent
from leoai.rag_context_utils import get_base_context
from main_config import (
    HOSTNAME, LEOBOT_DEV_MODE, REDIS_CLIENT, RATE_LIMIT_WINDOW_SECONDS,
    RATE_LIMIT_MAX_MESSAGES, BASE_URL_FB_MSG, FB_PAGE_ACCESS_TOKEN, FB_VERIFY_TOKEN,
    GEMINI_API_KEY, RESOURCES_DIR, TEMPLATES_DIR,
    ZALO_OA_ACCESS_TOKEN, get_current_user, leobot_lifespan
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

leobot = FastAPI(lifespan=leobot_lifespan)
rag_agent = RAGAgent()

# ===== CORS =====
origins = ["*"]
leobot.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()
@app.get("/ping")
def ping():
    return {"status": "ok", "msg": "Leo Bot running"}

# ===== Static Files & Templates =====
leobot.mount("/resources", StaticFiles(directory=RESOURCES_DIR), name="resources")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# === RATE LIMIT CHECK ===
def is_safe_to_answer(visitor_id: str) -> bool:
    key = f"chat_rate_limit:{visitor_id}"
    now = int(time.time() * 1000)
    window_start = now - (RATE_LIMIT_WINDOW_SECONDS * 1000)
    REDIS_CLIENT.zremrangebyscore(key, 0, window_start)
    current_count = REDIS_CLIENT.zcard(key)
    if current_count >= RATE_LIMIT_MAX_MESSAGES:
        return False
    REDIS_CLIENT.zadd(key, {str(now): now})
    REDIS_CLIENT.expire(key, RATE_LIMIT_WINDOW_SECONDS)
    return True


# ====== READY CHECK ======
@leobot.get("/_leoai/is-ready", response_class=JSONResponse)
@leobot.post("/_leoai/is-ready", response_class=JSONResponse)
async def is_leobot_ready():
    isReady = bool(GEMINI_API_KEY and GEMINI_API_KEY.strip())
    return {"ok": isReady}


# ===== FACEBOOK WEBHOOK =====
@leobot.get("/_leoai/fb-webhook", response_class=JSONResponse)
async def verify_fb_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == FB_VERIFY_TOKEN:
        logger.info("✅ FB Webhook verified.")
        return PlainTextResponse(content=challenge)
    else:
        logger.warning("❌ FB Webhook verification failed.")
        return JSONResponse(status_code=403, content={"error": "Verification failed"})


@leobot.post("/_leoai/fb-webhook", response_class=JSONResponse)
async def fb_webhook_handler(request: Request):
    try:
        body = await request.json()
        for entry in body.get("entry", []):
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]
                message_data = event.get("message", {})
                user_msg = message_data.get("text")
                if not user_msg:
                    continue
                logger.info(f"FB user '{sender_id}' said: {user_msg}")

                ai_reply = await rag_agent.process_chat_message(
                    user_id=sender_id,
                    user_message=user_msg,
                    persona_id="fb_user",
                    cdp_profile_id='',
                    touchpoint_id="facebook"
                )
                logger.info(f"AI reply to FB '{sender_id}': {ai_reply}")
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
        logger.info(f"✅ Sent message to FB user {recipient_id}")
    except Exception as e:
        logger.error(f"❌ Failed to send message to FB user {recipient_id}: {e}")


# === ZALO OA SUPPORT ===
@leobot.post("/_leoai/zalo-webhook", response_class=JSONResponse)
async def zalo_webhook_handler(request: Request):
    """
    Handle Zalo Official Account webhook events.
    """
    try:
        body = await request.json()
        logger.info(f"Zalo webhook body: {json.dumps(body)}")

        event_name = body.get("event_name")
        if event_name == "user_send_text":
            user_id = body["sender"]["id"]
            user_msg = body["message"]["text"]
            logger.info(f"Zalo user '{user_id}' said: {user_msg}")

            ai_reply = await rag_agent.process_chat_message(
                user_id=user_id,
                user_message=user_msg,
                persona_id="zalo_user",
                cdp_profile_id='',
                touchpoint_id="zalo"
            )
            logger.info(f"AI reply to Zalo '{user_id}': {ai_reply}")
            send_message_to_zalo(user_id, ai_reply)
    except Exception as e:
        logger.exception("Error in Zalo webhook handler")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"ok": True}


def send_message_to_zalo(recipient_id: str, message_text: str):
    """
    Send message to user via Zalo OA API
    """
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"user_id": recipient_id},
        "message": {"text": message_text}
    }
    params = {"access_token": ZALO_OA_ACCESS_TOKEN}
    try:
        response = requests.post(url, headers=headers, params=params, json=payload)
        response.raise_for_status()
        logger.info(f"✅ Sent message to Zalo user {recipient_id}")
    except Exception as e:
        logger.error(f"❌ Failed to send message to Zalo user {recipient_id}: {e}")


# ===== BASIC ROUTES =====
@leobot.get("/_leoai/ping", response_class=PlainTextResponse)
@leobot.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@leobot.get("/", response_class=HTMLResponse)
@leobot.get("/_leoai", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, 'timestamp': ts}
    return templates.TemplateResponse("index.html", data)


@leobot.get("/_leoai/demo-chatbot-ishop", response_class=HTMLResponse)
async def demo_chat_in_ishop(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, 'timestamp': ts}
    return templates.TemplateResponse("demo-chatbot-ishop.html", data)


@leobot.get("/_leoai/visitor-info", response_class=JSONResponse)
async def get_visitor_info(
    visitor_id: str = Query(..., description="Required visitor unique ID"),
    name: str | None = Query(None, description="Optional visitor name"),
    touchpoint_id: str | None = Query(None, description="Optional touchpoint ID"),
):
    # --- Validate GEMINI_API_KEY ---
    if not isinstance(GEMINI_API_KEY, str) or not GEMINI_API_KEY.strip():
        return JSONResponse(
            content={"answer": "GEMINI_API_KEY is empty", "error": True, "error_code": 501},
            status_code=500,
        )

    # --- Validate visitor_id ---
    visitor_id = visitor_id.strip()
    if not visitor_id:
        return JSONResponse(
            content={"answer": "visitor_id is empty", "error": True, "error_code": 500},
            status_code=400,
        )

    # --- Retrieve cached info ---
    redis_data = REDIS_CLIENT.hgetall(visitor_id)
    logger.info(redis_data)
    
    cached_name = redis_data.get("name","") 
    cached_touchpoint = redis_data.get("init_touchpoint_id","")

    # --- Determine final values ---
    final_name = name or cached_name or ""
    final_touchpoint = touchpoint_id or cached_touchpoint or ""
    
    logger.info(f"cached_name : {cached_name} visitor_id : {visitor_id}")
    logger.info(f"final_name : {final_name} final_touchpoint : {final_touchpoint}")

    # --- Cache new values if provided ---
    updates = {}
    if name and name != cached_name:
        updates["name"] = name
    if touchpoint_id and touchpoint_id != cached_touchpoint:
        updates["init_touchpoint_id"] = touchpoint_id

    if updates:
        REDIS_CLIENT.hset(visitor_id, mapping=updates)

    # --- Return final result ---
    return JSONResponse(
        content={
            "visitor_id": visitor_id,
            "name": final_name,
            "init_touchpoint_id": final_touchpoint,
            "cached": bool(redis_data),
            "error": False,
            "error_code": 0,
        },
        status_code=200,
    )


# ===== MAIN CHATBOT API =====
@leobot.post("/_leoai/ask", response_class=JSONResponse)
async def web_handler(msg: Message):
    visitor_id = msg.visitor_id
    if len(visitor_id) == 0:
        return {"answer": "visitor_id is empty ", "error": True, "error_code": 500}
    question = msg.question
    if len(question) > 1000:
        return {"answer": "Question must be less than 1000 characters!", "error": True, "error_code": 510}

    profile_id = REDIS_CLIENT.hget(visitor_id, 'profile_id')
    safe = is_safe_to_answer(visitor_id)
    logger.info(f"profile_id {profile_id} safe {safe} visitor_id {visitor_id}")

    if safe:
        answer = await rag_agent.process_chat_message(
            user_id=visitor_id,
            user_message=question,
            persona_id=msg.persona_id,
            cdp_profile_id=profile_id,
            touchpoint_id=msg.touchpoint_id,
            target_language=msg.answer_in_language,
            answer_in_format=msg.answer_in_format
        )
        logger.info(f"Answer for visitor {visitor_id}: {answer}")
        return {"question": question, "answer": answer, "visitor_id": visitor_id, "error_code": 0}
    else:
        logger.warning(f"Rate limit exceeded for visitor_id: {visitor_id}")
        return {"answer": "You are sending messages too quickly. Please wait a moment.", "error": True, "error_code": 429}
    



@leobot.post("/_leoai/email-agent", response_class=JSONResponse)
async def email_agent(request: Request, user: str = Depends(get_current_user)):
    try:
        # Parse JSON input
        data = await request.json()
        
        # Initialize EmailSender (with your SMTP credentials)
        agent = EmailSender()

        # Merge incoming JSON into base context
        context = get_base_context(data)
    
        # Extract email params safely
        to_email = data.get("to_email") or context.get("user_profile", {}).get("primary_email")
        subject = data.get("subject") or "AI Agent Email"
        template_name = data.get("template_name") or "welcome_email.html"

        if not to_email:
            return JSONResponse(status_code=400, content={"error": "Recipient email not provided"})

        success = await agent.send(
            to_email=to_email,
            subject=subject,
            template_name=template_name,
            context=context
        )

        if not success:
            return JSONResponse(status_code=500, content={"error": "Failed to send email"})

        return {"ok": True, "message": "Email sent successfully"}

    except Exception as e:
        logger.exception("Error in /email-agent handler")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
