import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# --- General ---
VERSION = "1.0.0"
SERVICE_NAME = f"LEO BOT VERSION: {VERSION}"
LEOBOT_DEV_MODE = os.getenv("LEOBOT_DEV_MODE") == "true"
HOSTNAME = os.getenv("HOSTNAME", "localhost")

# --- Rate Limiting ---
RATE_LIMIT_MAX_MESSAGES = 20  # max messages
RATE_LIMIT_WINDOW_SECONDS = 60  # time window

# --- Redis ---
REDIS_HOST = os.getenv("REDIS_USER_SESSION_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_USER_SESSION_PORT", 6379))

# --- Facebook Integration ---
BASE_URL_FB_MSG = 'https://graph.facebook.com/v13.0/me/messages'
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN")

# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- File Paths ---
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
TEMPLATES_DIR = RESOURCES_DIR / "templates"