#!/bin/bash

# === CONFIGURATION ===
LOG_DIR="."  # Adjust if logs are stored elsewhere
PATTERN="leobot-[0-9]{4}-[0-9]{2}-[0-9]{2}.log"

# === MOVE TO LOG DIRECTORY ===
if [ -d "$LOG_DIR" ]; then
  cd "$LOG_DIR"
else
  echo "Log directory $LOG_DIR does not exist."
  exit 1
fi

# === DELETE MATCHING LOG FILES ===
echo "Deleting logs matching pattern: $PATTERN"
find . -type f -regextype posix-extended -regex "./$PATTERN" -exec rm -v {} \;

echo "Cleanup complete."
