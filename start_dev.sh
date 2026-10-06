#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# LEO BOT — Improved Parallel Development Startup Script
# ------------------------------------------------------------------------------

# Colors for readable logs
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
BLUE="\033[1;34m"
NC="\033[0m"

echo -e "${GREEN}🚀 Starting LEO BOT dev environment...${NC}"

SEED_DATA=false
for arg in "$@"; do
  case "$arg" in
    --seed-data)
      SEED_DATA=true
      ;;
    *)
      echo -e "${RED}❌ Unknown option: $arg${NC}"
      exit 1
      ;;
  esac
done

# ------------------------------------------------------------------------------
# STEP 0: Load .env configuration
# ------------------------------------------------------------------------------
if [ -f .env ]; then
  echo -e "${BLUE}📄 Loading configuration from .env...${NC}"
  # 'set -a' automatically exports variables defined in the source file
  set -a
  source .env
  set +a
else
  echo -e "${RED}⚠️  Warning: .env file not found. Using script defaults.${NC}"
fi

# ------------------------------------------------------------------------------
# Configuration (Defaults can be overridden by .env)
# ------------------------------------------------------------------------------
PG_PORT="${PG_PORT:-5433}"
PG_WAIT_MAX=30

# Keycloak Config
KEYCLOAK_ENABLED="${KEYCLOAK_ENABLED:-false}" # Default to false if missing
KEYCLOAK_REALM="${KEYCLOAK_REALM:-master}"
KEYCLOAK_URL="${KEYCLOAK_URL:-https://leoid.example.com}"
KEYCLOAK_HEALTHCHECK="${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/.well-known/openid-configuration"

# App Config
VENV_PATH="env/bin/activate"
FASTAPI_APP="main_app:leobot"
FASTAPI_PORT="${FASTAPI_PORT:-8888}"

# ------------------------------------------------------------------------------
# Function: wait for PostgreSQL
# ------------------------------------------------------------------------------
wait_for_postgres() {
  echo -e "${YELLOW}🔍 Checking PostgreSQL on port ${PG_PORT}...${NC}"

  if nc -z localhost "$PG_PORT" 2>/dev/null; then
    echo -e "${GREEN}✅ PostgreSQL already running.${NC}"
    bash ./dockers/pgsql/start_pgsql_pgvector.sh
    return 0
  fi

  echo -e "${YELLOW}⚙️  Starting PostgreSQL docker (pgvector)...${NC}"
  # Ensure this path exists relative to where you run the script
  bash ./dockers/pgsql/start_pgsql_pgvector.sh

  for ((i=1; i<=PG_WAIT_MAX; i++)); do
    if nc -z localhost "$PG_PORT" 2>/dev/null; then
      echo -e "${GREEN}✅ PostgreSQL is now up (after ${i}s).${NC}"
      return 0
    fi
    sleep 1
  done

  echo -e "${RED}❌ PostgreSQL did not start after ${PG_WAIT_MAX} seconds.${NC}"
  exit 1
}

# ------------------------------------------------------------------------------
# Function: wait for Keycloak
# ------------------------------------------------------------------------------
wait_for_keycloak() {
  echo -e "${YELLOW}🔍 Checking Keycloak health at: ${KEYCLOAK_HEALTHCHECK}${NC}"

  # Quick check if already up
  HTTP_STATUS=$(curl -sk --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$KEYCLOAK_HEALTHCHECK" || true)

  if [[ "$HTTP_STATUS" == "200" ]]; then
    echo -e "${GREEN}✅ Keycloak is healthy.${NC}"
    return 0
  fi

  echo -e "${YELLOW}⚙️  Starting Keycloak Docker...${NC}"
  bash ./dockers/keycloak/start_keycloak.sh

  # Wait loop
  for i in {1..40}; do
    HTTP_STATUS=$(curl -sk --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$KEYCLOAK_HEALTHCHECK" || true)
    if [[ "$HTTP_STATUS" == "200" ]]; then
      echo -e "${GREEN}✅ Keycloak is now healthy (after ${i}s).${NC}"
      return 0
    fi
    sleep 1
  done

  echo -e "${RED}❌ Keycloak failed to respond after startup. Check docker logs.${NC}"
  exit 1
}

# ------------------------------------------------------------------------------
# EXECUTION FLOW
# ------------------------------------------------------------------------------

# 1. Start Postgres (Always required)
wait_for_postgres

if [[ "$SEED_DATA" == true ]]; then
  echo -e "${BLUE}🌱 --seed-data enabled. Loading sample places...${NC}"
  bash ./start-seeding-data.sh
fi

# 2. Start Keycloak (Only if enabled in .env)
if [[ "$KEYCLOAK_ENABLED" == "true" ]]; then
  echo -e "${BLUE}ℹ️  KEYCLOAK_ENABLED is true. Initializing Keycloak...${NC}"
  wait_for_keycloak
else
  echo -e "${BLUE}ℹ️  KEYCLOAK_ENABLED is '$KEYCLOAK_ENABLED'. Skipping Keycloak startup.${NC}"
fi

# 3. Update Git
echo -e "${YELLOW}📦 Updating Git repository...${NC}"
git pull --quiet
echo -e "${GREEN}✅ Repository updated.${NC}"

# 4. Activate Venv
if [[ -f "$VENV_PATH" ]]; then
  echo -e "${YELLOW}🐍 Activating Python virtual environment...${NC}"
  source "$VENV_PATH"
else
  echo -e "${RED}❌ Virtual environment not found at $VENV_PATH.${NC}"
  exit 1
fi

PYTHON_MINOR="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_MINOR" != "3.12" ]]; then
  echo -e "${RED}❌ Python 3.12 is required, but the active environment uses Python ${PYTHON_MINOR}.${NC}"
  exit 1
fi
echo -e "${GREEN}✅ Using Python ${PYTHON_MINOR}.${NC}"

# 5. Run FastAPI
echo -e "${YELLOW}⚡ Launching FastAPI (port ${FASTAPI_PORT})...${NC}"
uvicorn "$FASTAPI_APP" \
  --reload \
  --env-file .env \
  --host 0.0.0.0 \
  --port "$FASTAPI_PORT"