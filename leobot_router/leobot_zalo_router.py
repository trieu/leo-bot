import logging
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from leoai.rag_agent import RAGAgent
from main_config import ZALO_OA_ACCESS_TOKEN

# Zalo webhook router
logger = logging.getLogger("leobot-zalo")
router = APIRouter()
rag_agent = RAGAgent()


@router.post("/_leoai/zalo-webhook", response_class=JSONResponse)
async def zalo_webhook_handler(request: Request):
    """
    Zalo Official Account webhook for receiving messages.
    Processes text events and replies with AI-generated messages.
    """
    try:
        body = await request.json()
        event_name = body.get("event_name")

        if event_name == "user_send_text":
            user_id = body["sender"]["id"]
            user_msg = body["message"]["text"]
            logger.info(f"Zalo user '{user_id}' said: {user_msg}")

            ai_reply = await rag_agent.process_chat_message(
                user_id=user_id,
                user_message=user_msg,
                persona_id="zalo_user",
                cdp_profile_id="",
                touchpoint_id="zalo",
            )
            send_message_to_zalo(user_id, ai_reply)
    except Exception as e:
        logger.exception("Error in Zalo webhook handler")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"ok": True}


def send_message_to_zalo(recipient_id: str, message_text: str):
    """
    Sends a message to a Zalo user via the OA API.
    """
    url = "https://openapi.zalo.me/v3.0/oa/message/cs"
    params = {"access_token": ZALO_OA_ACCESS_TOKEN}
    payload = {"recipient": {"user_id": recipient_id}, "message": {"text": message_text}}

    try:
        with httpx.Client(timeout=10) as client:
            res = client.post(url, params=params, json=payload)
            res.raise_for_status()
            data = res.json()
            if data.get("error") == 0:
                logger.info(f"✅ Sent message to Zalo user {recipient_id}")
            else:
                logger.warning(f"⚠️ Zalo API returned error: {data}")
    except Exception as e:
        logger.error(f"❌ Failed to send Zalo message to {recipient_id}: {e}")
