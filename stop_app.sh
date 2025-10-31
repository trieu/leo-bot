#!/bin/sh

kill -15 $(pgrep -f "uvicorn main_app:leobot")
sleep 2