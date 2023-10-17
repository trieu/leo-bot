import google.generativeai as palm
from transformers import pipeline

from fastapi import FastAPI
from redis import Redis
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import os


REDIS_USER_SESSION_HOST = os.getenv("REDIS_USER_SESSION_HOST")
REDIS_USER_SESSION_PORT = os.getenv("REDIS_USER_SESSION_PORT")

GOOGLE_API_KEY = os.getenv("GOOGLE_PALM_API_KEY")
TEMPERATURE_SCORE = 0.7

MODEL_NLP = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NLP, tokenizer=MODEL_NLP)

VERSION = "0.0.1"
SERVICE_NAME = "LEO BOT VERSION:" + VERSION

# init FAST API app
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init API KEY
palm.configure(api_key=GOOGLE_API_KEY)

# Redis
r = Redis(host=REDIS_USER_SESSION_HOST,
          port=REDIS_USER_SESSION_PORT, decode_responses=True)


# Data models

class Message(BaseModel):
    content: str
    prompt: Optional[str] = Field(None, description="Who sends the error message.")
    usersession: str
    userlogin: str

def buildPrompt(promptText):
    p = {'prompt' : promptText, 'temperature' : TEMPERATURE_SCORE}
    return p

# API handlers

@app.get("/")
async def root():
    return {"SERVICE_NAME": SERVICE_NAME}


@app.post("/ask")
async def ask(question: Message):
    content = question.content
    print("ask question "+content)
    print(question.usersession)
    userLogin = r.hget(question.usersession, 'userlogin')
    print(userLogin)
    if userLogin == question.userlogin:
        # answer = llm(question.content)
        answer = "My name is LEO. That's a great question, but I don't have the answer right now. I'll do some research and get back to you."
        try:
            response = palm.generate_text(**buildPrompt(question.prompt))
            if response is not None and isinstance(response.result, str):
                answer = response.result
        except Exception as error:
            print("An exception occurred:", error)
        
        print("answer " + answer)
        data = {"question": content, "answer": answer, "userLogin": userLogin}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return 


@app.post("/sentiment-analysis")
async def sentiment_analysis(msg: Message):
    content = msg.content
    print("sentiment_analysis msg "+content)
    userLogin = r.hget(msg.usersession, 'userlogin')
    data = {"error": True}
    if userLogin == msg.userlogin:
        try:
            rs = sentiment_pipe(content)
            if len(rs) > 0 :
                data = rs[0]
        except Exception as error:
            print("An exception occurred:", error)
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data


@app.get("/is-ready")
async def is_leobot_ready():
    isReady = isinstance(GOOGLE_API_KEY, str)
    return {"ok": isReady }

@app.get("/ping")
async def ping():
    return "PONG"
