import os
import time
import logging
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi_keycloak import FastAPIKeycloak
from main_config import HOSTNAME, LEOBOT_DEV_MODE, TEMPLATES_DIR

logger = logging.getLogger("leobot-admin")

router = APIRouter()
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ===== Keycloak Setup =====
keycloak_enabled = os.getenv("KEYCLOAK_ENABLED", "true").lower() in ["true", "1", "yes"]
keycloak_openid = None

if not keycloak_enabled:
    logger.warning("🚫 Keycloak disabled (KEYCLOAK_ENABLED=false). Skipping authentication setup.")
else:
    verify_ssl_env = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower()
    verify_ssl = verify_ssl_env not in ["false", "0", "no"]

    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger.warning("⚠️ SSL verification disabled for Keycloak (development mode).")

    try:
        keycloak_openid = FastAPIKeycloak(
            server_url=os.getenv("KEYCLOAK_URL", "https://leoid.example.com"),
            realm=os.getenv("KEYCLOAK_REALM", "master"),
            client_id=os.getenv("KEYCLOAK_CLIENT_ID", "leobot-admin"),
            client_secret=os.getenv("KEYCLOAK_CLIENT_SECRET"),
            admin_client_secret=os.getenv("KEYCLOAK_CLIENT_SECRET"),
            callback_uri=os.getenv(
                "KEYCLOAK_CALLBACK_URL",
                "https://leobot.example.com/_leoai/sso/callback"
            ),
        )

        # Disable SSL verification internally if needed
        if not verify_ssl:
            if hasattr(keycloak_openid, "keycloak_openid"):
                keycloak_openid.keycloak_openid.connection.verify = False
            if hasattr(keycloak_openid, "keycloak_admin"):
                keycloak_openid.keycloak_admin.connection.verify = False

        logger.info("✅ Keycloak client initialized successfully.")

    except Exception as e:
        logger.exception(f"❌ Failed to initialize Keycloak client: {e}")
        keycloak_openid = None


# ===== Health Checks =====
@router.get("/_leoai/ping", response_class=PlainTextResponse)
@router.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@router.get("/_leoai/is-ready", response_class=JSONResponse)
async def is_ready():
    return {"ok": keycloak_openid is not None}


# ===== Public Root =====
@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    context = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, "timestamp": ts}
    return templates.TemplateResponse("index.html", context)


# ===== Admin Panel =====
if keycloak_openid:
    @router.get("/admin", response_class=HTMLResponse)
    @router.get("/_leoai/admin", response_class=HTMLResponse)
    async def admin_panel(request: Request, user=Depends(keycloak_openid.get_current_user(required=True))):
        ts = int(time.time())
        user_info = {
            "username": user.get("preferred_username"),
            "email": user.get("email"),
            "name": user.get("name"),
        }
        context = {
            "request": request,
            "HOSTNAME": HOSTNAME,
            "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE,
            "timestamp": ts,
            "user": user_info,
        }
        return templates.TemplateResponse("admin/dashboard.html", context)


    # ===== Keycloak OAuth2 flow =====
    @router.get("/_leoai/sso/callback", response_class=HTMLResponse)
    async def keycloak_callback(request: Request):
        token = await keycloak_openid.exchange_code_for_token(request)
        response = RedirectResponse(url="/admin")
        keycloak_openid.set_token_cookies(response, token)
        return response


    @router.get("/_leoai/logout")
    async def logout(request: Request):
        response = RedirectResponse(url="/")
        keycloak_openid.logout(request, response)
        return response
