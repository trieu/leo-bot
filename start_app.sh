#!/bin/bash

DIR_PATH="/build/leo-bot/"

# Change to the directory where your FastAPI app is located

if [ -d "$DIR_PATH" ]; then
  cd $DIR_PATH
fi

# Activate your virtual environment if necessary
SOURCE_PATH="env/bin/activate"
source $SOURCE_PATH

# clear old log
cat /dev/null > leobot.log

# Start the FastAPI app using uvicorn
uvicorn main:app --reload --env-file .env --host 0.0.0.0 --port 8888 >> leobot.log 2>&1 &

# exit
deactivate