import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# configs
from main_config import RESOURCES_DIR, TEMPLATES_DIR, leobot_lifespan

# inbound router
from leobot_router.leobot_main_router import router as leobot_main_router
from leobot_router.leobot_assets_router import router as leobot_assets_router

# outbound router
from leobot_router.leobot_email_router import router as leobot_email_router
from leobot_router.leobot_facebook_router import router as leobot_facebook_router
from leobot_router.leobot_zalo_router import router as leobot_zalo_router

# admin router
from leobot_router.leobot_admin_router import keycloak_enabled, router as leobot_admin_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_app")

def create_app() -> FastAPI:
    app = FastAPI(lifespan=leobot_lifespan)

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static & templates
    app.mount("/resources", StaticFiles(directory=RESOURCES_DIR), name="resources")
    app.state.templates = Jinja2Templates(directory=TEMPLATES_DIR)

    # Routers
    app.include_router(leobot_main_router)
    app.include_router(leobot_assets_router)
    app.include_router(leobot_facebook_router)
    app.include_router(leobot_zalo_router)
    app.include_router(leobot_email_router) 

    # Conditionally add admin router
    if keycloak_enabled:
        logger.info("🔐 Keycloak enabled: mounting admin routes.")
        app.include_router(leobot_admin_router)
    else:
        logger.warning("🚫 Keycloak disabled: skipping admin routes.")

    return app


# leo bot app
leobot = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_app:leobot", host="0.0.0.0", port=8888, reload=True)