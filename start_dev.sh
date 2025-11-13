#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# LEO BOT — Parallel Development Startup Script
# ------------------------------------------------------------------------------
# Fast, fault-tolerant startup sequence:
#   1. Checks PostgreSQL (5432) and Keycloak (via Nginx HTTPS realm endpoint).
#   2. Starts any missing services in parallel.
#   3. Updates code, activates venv, and runs FastAPI app.
# ------------------------------------------------------------------------------

PG_PORT=5432
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

echo -e "${GREEN}🚀 Starting LEO BOT dev environment (parallel mode)...${NC}"

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

check_postgres() {
  if nc -z localhost "$PG_PORT" 2>/dev/null; then
    echo -e "${GREEN}✅ PostgreSQL already running on port $PG_PORT.${NC}"
  else
    echo -e "${YELLOW}⚙️  Starting PostgreSQL (pgvector)...${NC}"
    bash ./dockers/pgsql/start_pgsql_pgvector.sh
    for i in {1..5}; do
      sleep 1
      if nc -z localhost "$PG_PORT" 2>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL is now up.${NC}"
        return
      fi
    done
    echo -e "${RED}❌ PostgreSQL failed to start on port $PG_PORT.${NC}"
    exit 1
  fi
}

check_keycloak() {
  echo -e "${YELLOW}🔍 Checking Keycloak health at:${NC}"
  echo -e "    ${KEYCLOAK_HEALTHCHECK}"
  HTTP_STATUS=$(curl -sk --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$KEYCLOAK_HEALTHCHECK")

  if [[ "$HTTP_STATUS" =~ ^(200)$ ]]; then
    echo -e "${GREEN}✅ Keycloak is healthy (HTTP $HTTP_STATUS).${NC}"
  else
    echo -e "${YELLOW}⚙️  Starting Keycloak (Docker)...${NC}"
    bash ./dockers/keycloak/run-keycloak.sh
    for i in {1..10}; do
      sleep 1
      HTTP_STATUS=$(curl -sk --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$KEYCLOAK_HEALTHCHECK")
      if [[ "$HTTP_STATUS" =~ ^(200)$ ]]; then
        echo -e "${GREEN}✅ Keycloak is now healthy (HTTP $HTTP_STATUS).${NC}"
        return
      fi
    done
    echo -e "${RED}❌ Keycloak failed to respond after startup. Check Docker logs.${NC}"
    exit 1
  fi
}

# ------------------------------------------------------------------------------
# Step 1–2: Parallel checks
# ------------------------------------------------------------------------------
check_postgres &
PID_PG=$!

check_keycloak &
PID_KC=$!

# Wait for both processes
wait $PID_PG
wait $PID_KC

# ------------------------------------------------------------------------------
# Step 3: Update repo
# ------------------------------------------------------------------------------
echo -e "${YELLOW}📦 Updating Git repository...${NC}"
git pull --quiet
echo -e "${GREEN}✅ Repository is up to date.${NC}"

# ------------------------------------------------------------------------------
# Step 4: Activate virtual environment
# ------------------------------------------------------------------------------
if [[ -f "$VENV_PATH" ]]; then
  echo -e "${YELLOW}🐍 Activating Python virtual environment...${NC}"
  source "$VENV_PATH"
else
  echo -e "${RED}❌ Virtual environment not found at $VENV_PATH.${NC}"
  exit 1
fi

# ------------------------------------------------------------------------------
# Step 5: Launch FastAPI app
# ------------------------------------------------------------------------------
echo -e "${YELLOW}⚡ Launching FastAPI app (port $FASTAPI_PORT)...${NC}"
uvicorn "$FASTAPI_APP" \
  --reload \
  --env-file .env \
  --host 0.0.0.0 \
  --port $FASTAPI_PORT