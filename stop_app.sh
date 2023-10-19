#!/bin/sh

kill -15 $(pgrep -f "uvicorn main:leobot")
sleep 2