#!/bin/sh

kill -15 $(pgrep -f uvicorn)
sleep 2