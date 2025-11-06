import os
import time
import logging
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_keycloak import FastAPIKeycloak
from main_config import HOSTNAME, LEOBOT_DEV_MODE, RESOURCES_DIR, TEMPLATES_DIR, leobot_lifespan

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leobot-admin")

# ===== FastAPI App =====
leobot = FastAPI(lifespan=leobot_lifespan)

# ===== CORS =====
leobot.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Static & Templates =====
leobot.mount("/resources", StaticFiles(directory=RESOURCES_DIR), name="resources")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ===== Keycloak Auth =====
keycloak_enabled = os.getenv("KEYCLOAK_ENABLED", "true").lower() in ["true", "1", "yes"]

if not keycloak_enabled:
    print("🚫 Keycloak disabled (KEYCLOAK_ENABLED=false). Skipping authentication setup.")
    keycloak_openid = None
else:
    verify_ssl_env = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower()
    verify_ssl = verify_ssl_env not in ["false", "0", "no"]

    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("⚠️  SSL verification disabled for Keycloak (development mode).")

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

        keycloak_openid.add_swagger_config(leobot)
        print("✅ Keycloak client initialized successfully.")

    except Exception as e:
        print(f"❌ Failed to initialize Keycloak client: {e}")
        keycloak_openid = None


# ===== Simple health checks =====
@leobot.get("/_leoai/ping", response_class=PlainTextResponse)
@leobot.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@leobot.get("/_leoai/is-ready", response_class=JSONResponse)
async def is_ready():
    return {"ok": keycloak_openid is not None}


# ===== Root & Admin Views =====
@leobot.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    context = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, "timestamp": ts}
    return templates.TemplateResponse("index.html", context)


@leobot.get("/admin", response_class=HTMLResponse)
@leobot.get("/_leoai/admin", response_class=HTMLResponse)
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
@leobot.get("/_leoai/sso/callback", response_class=HTMLResponse)
async def keycloak_callback(request: Request):
    token = await keycloak_openid.exchange_code_for_token(request)
    response = RedirectResponse(url="/admin")
    keycloak_openid.set_token_cookies(response, token)
    return response


@leobot.get("/_leoai/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/")
    keycloak_openid.logout(request, response)
    return response
