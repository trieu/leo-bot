import os
from transformers import pipeline

from fastapi import FastAPI
from redis import Redis
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from leoai import leo_chatbot

REDIS_USER_SESSION_HOST = os.getenv("REDIS_USER_SESSION_HOST")
REDIS_USER_SESSION_PORT = os.getenv("REDIS_USER_SESSION_PORT")

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

# Redis
r = Redis(host=REDIS_USER_SESSION_HOST, port=REDIS_USER_SESSION_PORT, decode_responses=True)

# Data models

class Message(BaseModel):
    answer_in_language: Optional[str] = Field("vi")
    content: str
    prompt: Optional[str] = Field(None, description="the question")
    usersession: str
    userlogin: str

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

        # our model can only understand English
        lang = question.answer_in_language
        question_in_english = content
        if lang != "en":
            question_in_english = leo_chatbot.translate_text('en',content) 
        # translate if need
        answer = leo_chatbot.ask_question(lang, question_in_english)

        print("answer " + answer)
        data = {"question": content, "answer": answer, "userLogin": userLogin}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data


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
    isReady = isinstance(leo_chatbot.GOOGLE_GENAI_API_KEY, str)
    return {"ok": isReady }

@app.get("/ping")
async def ping():
    return "PONG"
