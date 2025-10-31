#!/usr/bin/env bash
set -euo pipefail

APP_USER="leocdp"
APP_DIR="/build/leo-bot"
RESTART_SCRIPT="$APP_DIR/restart_app.sh"
CHECK_URL="http://127.0.0.1:8888/ping"
LOG_DIR="$APP_DIR/cron-jobs"
LOG_FILE="$LOG_DIR/healthcheck.log"
TIMEOUT=5

mkdir -p "$LOG_DIR"

# Timestamp function for logging
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$LOG_FILE"
}

# Health check
response_code=$(curl -m $TIMEOUT -s -o /dev/null -w "%{http_code}" "$CHECK_URL" || echo "000")

if [ "$response_code" -eq 200 ]; then
  log "✅ Healthy (HTTP 200)"
  exit 0
fi

log "❌ Unhealthy (HTTP $response_code). Restarting..."

# Wait a bit before restart
sleep 2

if [[ ! -f "$RESTART_SCRIPT" ]]; then
  log "⛔ Restart script not found: $RESTART_SCRIPT"
  exit 1
fi

# Restart the app as the correct user
if sudo -u "$APP_USER" bash "$RESTART_SCRIPT"; then
  log "🔁 Restart triggered successfully."
else
  log "🔥 Restart FAILED!"
  exit 1
fi
