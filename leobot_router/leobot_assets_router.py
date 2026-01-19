import logging
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from main_config import RESOURCES_DIR

logger = logging.getLogger("leobot-assets")

router = APIRouter()
ICONS_DIR = Path(RESOURCES_DIR) / "icons"


def serve_file(path: Path, media_type: str, name: str):
    """Helper to safely serve static asset files."""
    if path.exists():
        return FileResponse(path, media_type=media_type)
    logger.warning(f"⚠️ Missing asset: {name} at {path}")
    return JSONResponse(status_code=404, content={"error": f"{name} not found"})


@router.get("/favicon.ico")
async def favicon():
    """Serve favicon.ico"""
    return serve_file(ICONS_DIR / "favicon.ico", "image/x-icon", "favicon.ico")


@router.get("/apple-touch-icon.png")
async def apple_touch_icon():
    """Serve Apple touch icon (iOS home screen icon)."""
    return serve_file(ICONS_DIR / "apple-touch-icon.png", "image/png", "apple-touch-icon.png")


@router.get("/favicon-32x32.png")
async def favicon_32():
    """Serve 32x32 favicon."""
    return serve_file(ICONS_DIR / "favicon-32x32.png", "image/png", "favicon-32x32.png")


@router.get("/favicon-16x16.png")
async def favicon_16():
    """Serve 16x16 favicon."""
    return serve_file(ICONS_DIR / "favicon-16x16.png", "image/png", "favicon-16x16.png")


@router.get("/site.webmanifest")
async def manifest():
    """Serve web app manifest (for PWA metadata)."""
    return serve_file(ICONS_DIR / "site.webmanifest", "application/manifest+json", "site.webmanifest")
