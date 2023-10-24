#!/bin/bash

git pull

# Activate your virtual environment if necessary
SOURCE_PATH="env/bin/activate"
source $SOURCE_PATH

# Start the FastAPI app using uvicorn
uvicorn main:leobot --reload --env-file .env --host 0.0.0.0 --port 8888 