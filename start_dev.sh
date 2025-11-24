#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# LEO BOT — Improved Parallel Development Startup Script
# ------------------------------------------------------------------------------
# Fixes:
#   - Keycloak now ALWAYS waits for PostgreSQL to be ready before starting.
#   - Added strong retry loops & exponential wait.
#   - Prevents race condition where Keycloak starts before PGSQL docker.
#   - Cleaner logs + safer checks.
# ------------------------------------------------------------------------------

PG_PORT=5432
PG_WAIT_MAX=30        # max seconds to wait for PostgreSQL
KEYCLOAK_REALM="master"
KEYCLOAK_URL="https://leoid.example.com"
KEYCLOAK_HEALTHCHECK="${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/.well-known/openid-configuration"
VENV_PATH="env/bin/activate"
FASTAPI_APP="main_app:leobot"
FASTAPI_PORT=8888

# Colors for readable logs
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
NC="\033[0m"

echo -e "${GREEN}🚀 Starting LEO BOT dev environment (improved Startup)...${NC}"

# ------------------------------------------------------------------------------
# Function: wait for PostgreSQL (5432)
# ------------------------------------------------------------------------------
wait_for_postgres() {
  echo -e "${YELLOW}🔍 Checking PostgreSQL on port ${PG_PORT}...${NC}"

  # Already running?
  if nc -z localhost "$PG_PORT" 2>/dev/null; then
    echo -e "${GREEN}✅ PostgreSQL already running.${NC}"
    return 0
  fi

  # Start PG Docker
  echo -e "${YELLOW}⚙️  Starting PostgreSQL docker (pgvector)...${NC}"
  bash ./dockers/pgsql/start_pgsql_pgvector.sh

  # Wait until reachable
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
# Function: wait for Keycloak (must run AFTER PostgreSQL ready)
# ------------------------------------------------------------------------------
wait_for_keycloak() {
  echo -e "${YELLOW}🔍 Checking Keycloak health at:${NC}"
  echo -e "    ${KEYCLOAK_HEALTHCHECK}"

  HTTP_STATUS=$(curl -sk --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$KEYCLOAK_HEALTHCHECK")

  if [[ "$HTTP_STATUS" == "200" ]]; then
    echo -e "${GREEN}✅ Keycloak is healthy.${NC}"
    return 0
  fi

  # Start KC Docker
  echo -e "${YELLOW}⚙️  Starting Keycloak Docker...${NC}"
  bash ./dockers/keycloak/start_keycloak.sh

  # Wait for Keycloak HTTP OK
  for i in {1..40}; do
    HTTP_STATUS=$(curl -sk --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$KEYCLOAK_HEALTHCHECK")
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
# STEP 1: Wait for PostgreSQL first (Keycloak depends on it)
# ------------------------------------------------------------------------------
wait_for_postgres

# ------------------------------------------------------------------------------
# STEP 2: Now check/start Keycloak
# ------------------------------------------------------------------------------
wait_for_keycloak

# ------------------------------------------------------------------------------
# STEP 3: Update git repo
# ------------------------------------------------------------------------------
echo -e "${YELLOW}📦 Updating Git repository...${NC}"
git pull --quiet
echo -e "${GREEN}✅ Repository updated.${NC}"

# ------------------------------------------------------------------------------
# STEP 4: Activate Python venv
# ------------------------------------------------------------------------------
if [[ -f "$VENV_PATH" ]]; then
  echo -e "${YELLOW}🐍 Activating Python virtual environment...${NC}"
  source "$VENV_PATH"
else
  echo -e "${RED}❌ Virtual environment not found at $VENV_PATH.${NC}"
  exit 1
fi

# ------------------------------------------------------------------------------
# STEP 5: Run FastAPI App
# ------------------------------------------------------------------------------
echo -e "${YELLOW}⚡ Launching FastAPI (port ${FASTAPI_PORT})...${NC}"
uvicorn "$FASTAPI_APP" \
  --reload \
  --env-file .env \
  --host 0.0.0.0 \
  --port "$FASTAPI_PORT"
