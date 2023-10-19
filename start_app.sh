#!/bin/bash

DIR_PATH="/build/leo-bot/"

# Change to the directory where your FastAPI app is located

if [ -d "$DIR_PATH" ]; then
  cd $DIR_PATH
fi

# kill old process to restart
kill -15 $(pgrep -f "uvicorn main:leobot")
sleep 2

# Activate your virtual environment if necessary
SOURCE_PATH="env/bin/activate"
source $SOURCE_PATH

# clear old log
# cat /dev/null > leobot.log
datetoday=$(date '+%Y-%m-%d')
log_file="leobot-$datetoday.log"


# Start the FastAPI app using uvicorn
uvicorn main:leobot --reload --env-file .env --host 0.0.0.0 --port 8888 >> $log_file 2>&1 &

# exit
deactivate