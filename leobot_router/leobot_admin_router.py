import os
import time
import logging
from dotenv import load_dotenv
import urllib3

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi_keycloak import FastAPIKeycloak  # Import the class
from fastapi_keycloak.exceptions import KeycloakError # Import exception for clarity
from main_config import HOSTNAME, LEOBOT_DEV_MODE, TEMPLATES_DIR

logger = logging.getLogger("leobot-admin")

# Load environment variables from .env file
load_dotenv(override=True)

router = APIRouter()
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ===== Keycloak Setup =====
keycloak_enabled = os.getenv("KEYCLOAK_ENABLED", "true").lower() in ["true", "1", "yes"]
verify_ssl = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() not in ["false", "0", "no"]

if not verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

keycloak_openid = None

if keycloak_enabled:
    try:
        # FIX 1: Monkey-patch the class *before* initialization
        FastAPIKeycloak._get_admin_token = lambda *_, **__: None
        logger.info("Patched FastAPIKeycloak._get_admin_token to skip service account login.")

        # Now, we can safely initialize the instance.
        keycloak_openid = FastAPIKeycloak(
            server_url=os.getenv("KEYCLOAK_URL"),
            realm=os.getenv("KEYCLOAK_REALM"),
            client_id=os.getenv("KEYCLOAK_CLIENT_ID"),
            callback_uri=os.getenv("KEYCLOAK_CALLBACK_URL"),
            ssl_verification=verify_ssl,
            timeout=10,
            client_secret=None,        # Correct for public client
            admin_client_secret=None,  # Correct for no service account
        )
        
        logger.info("✅ Keycloak initialized for browser SSO (no admin API).")

    except Exception as e:
        if "unauthorized_client" in str(e):
             logger.error(f"❌ Keycloak initialization failed: {e}. Check if your Keycloak client is 'public' and 'Service Accounts Enabled' is OFF.")
        else:
            logger.exception(f"❌ Keycloak initialization failed: {e}")
else:
    logger.warning("🚫 Keycloak disabled in .env.")


# ===== Health Check =====
@router.get("/_leoai/is-admin-ready", response_class=JSONResponse)
async def is_ready():
    return {"ok": keycloak_openid is not None}


# ===== Protected Admin Page =====
if keycloak_openid:
    @router.get("/admin", response_class=HTMLResponse)
    @router.get("/_leoai/admin", response_class=HTMLResponse)
    async def admin_panel(
        request: Request,
        user=Depends(keycloak_openid.get_current_user())  # FIX 2: Removed 'required=True'
    ):
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

    # ===== OAuth2 Flow =====
    @router.get("/_leoai/login")
    async def login():
        return RedirectResponse(url=keycloak_openid.login_uri) # FIX 3: Removed parentheses

    @router.get("/_leoai/sso/callback", response_class=HTMLResponse)
    async def keycloak_callback(session_state: str, code: str): # FIX 4 (Part A)
        # FIX 4 (Part B): Correct method, new params, and no 'await'
        token = keycloak_openid.exchange_authorization_code(
            session_state=session_state,
            code=code
        )
        
        response = RedirectResponse(url="/admin")
        keycloak_openid.set_token_cookies(response, token)
        return response

    @router.get("/_leoai/logout")
    async def logout(request: Request):
        response = RedirectResponse(url="/")
        keycloak_openid.logout(request, response)
        return response

# Handle case where Keycloak is disabled or failed to init
else:
    @router.get("/admin", response_class=HTMLResponse)
    @router.get("/_leoai/admin", response_class=HTMLResponse)
    async def admin_panel_disabled(request: Request):
        context = {"request": request, "error_message": "Keycloak is not enabled or failed to initialize."}
        return templates.TemplateResponse("admin/error.html", context, status_code=503)

    @router.get("/_leoai/login")
    async def login_disabled():
        return JSONResponse(status_code=503, content={"error": "KeyScope (Keycloak) is not configured"})

    @router.get("/_leoai/logout")
    async def logout_disabled():
        return RedirectResponse(url="/")