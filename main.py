import os
from transformers import pipeline

from fastapi import FastAPI
from redis import Redis
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from leoai import leo_chatbot

LEOBOT_DEV_MODE = os.getenv("LEOBOT_DEV_MODE") == "true"
HOSTNAME = os.getenv("HOSTNAME")
REDIS_USER_SESSION_HOST = os.getenv("REDIS_USER_SESSION_HOST")
REDIS_USER_SESSION_PORT = os.getenv("REDIS_USER_SESSION_PORT")

MODEL_NLP = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NLP, tokenizer=MODEL_NLP)

# Redis
redis = Redis(host=REDIS_USER_SESSION_HOST, port=REDIS_USER_SESSION_PORT, decode_responses=True)

VERSION = "0.0.1"
SERVICE_NAME = "LEO BOT VERSION:" + VERSION
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/resources/"
MAIN_HTML_LEO_BOT = SERVICE_NAME
with open(os.path.join(ROOT_FOLDER, 'index.html')) as fh:
    MAIN_HTML_LEO_BOT = fh.read()
MAIN_HTML_LEO_BOT = MAIN_HTML_LEO_BOT.replace('$HOSTNAME', HOSTNAME)

# init FAST API leobot
leobot = FastAPI()
origins = ["*"]
leobot.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Message(BaseModel):
    answer_in_language: Optional[str] = Field("en") # default is English
    content: str
    prompt: Optional[str] = Field(None, description="the question")
    usersession: str
    userlogin: str

##### API handlers #####

@leobot.get("/", response_class=HTMLResponse)
async def root():
    if LEOBOT_DEV_MODE :
        index_html = ''
        with open(os.path.join(ROOT_FOLDER, 'index.html')) as fh:
            index_html = fh.read()
        index_html = index_html.replace('$HOSTNAME', HOSTNAME)
        return HTMLResponse(content=index_html, status_code=200)
    return HTMLResponse(content=MAIN_HTML_LEO_BOT, status_code=200)

@leobot.get("/resources/leocdp.chatbot.js", response_class=FileResponse)
async def chatbot_javascript():
    path = ROOT_FOLDER + "leocdp.chatbot.js"
    return FileResponse(path)

# the main API of chatbot
@leobot.post("/ask")
async def ask(question: Message):
    userLogin = redis.hget(question.usersession, 'userlogin')    
    content = question.content
    
    print("question.content: "+content)
    print("question.userlogin: "+question.userlogin)
    print("usersession: "+ question.usersession)    
    print("userLogin: "+userLogin)
    
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

@leobot.post("/sentiment-analysis")
async def sentiment_analysis(msg: Message):
    content = msg.content
    print("sentiment_analysis msg "+content)
    userLogin = redis.hget(msg.usersession, 'userlogin')
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

@leobot.get("/is-ready")
async def is_leobot_ready():
    isReady = isinstance(leo_chatbot.GOOGLE_GENAI_API_KEY, str)
    return {"ok": isReady }

@leobot.get("/ping")
async def ping():
    return "PONG"