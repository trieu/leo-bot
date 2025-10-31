#!/usr/bin/env bash
set -euo pipefail

APP_USER="leocdp"
APP_DIR="/build/leo-bot"
START_SCRIPT="$APP_DIR/start_app.sh"
CRON_LOG_DIR="$APP_DIR/cron-jobs"
CRON_LOG_FILE="$CRON_LOG_DIR/restart.log"

mkdir -p "$CRON_LOG_DIR"

{
  echo "──────────────────────────────────────────────"
  echo "📅 Restart triggered at: $(date '+%Y-%m-%d %H:%M:%S')"

  if [[ ! -f "$START_SCRIPT" ]]; then
    echo "❌ start-app.sh not found at $START_SCRIPT"
    exit 1
  fi

  echo "🔁 Running start_app.sh as $APP_USER..."
  sudo -u "$APP_USER" bash "$START_SCRIPT"

  echo "✅ Restart completed: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$CRON_LOG_FILE" 2>&1
