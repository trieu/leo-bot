#!/bin/bash

DIR_PATH="/home/thomas/0-uspa/leo-bot/"

# Change to the directory where your FastAPI app is located
cd $DIR_PATH

# Activate your virtual environment if necessary
source $DIR_PATH.venv/bin/activate

# Start the FastAPI app using uvicorn
uvicorn main:app --reload --env-file .env --host 0.0.0.0 --port 8888 