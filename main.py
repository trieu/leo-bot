from langchain.llms import OpenAI
from fastapi import FastAPI
from redis import Redis
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os

REDIS_USER_SESSION_HOST = os.getenv("REDIS_USER_SESSION_HOST")
REDIS_USER_SESSION_PORT = os.getenv("REDIS_USER_SESSION_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

# init LangChain
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# Redis
r = Redis(host=REDIS_USER_SESSION_HOST,
          port=REDIS_USER_SESSION_PORT, decode_responses=True)


# Data models

class Question(BaseModel):
    content: str
    usersession: str
    userlogin: str


# API handlers


@app.get("/")
async def root():
    return {"SERVICE_NAME": SERVICE_NAME}


@app.post("/ask")
async def ask(question: Question):
    content = question.content
    print(content)
    print(question.usersession)
    userLogin = r.hget(question.usersession, 'userlogin')
    print(userLogin)
    if userLogin == question.userlogin:
        answer = llm(question.content)
        data = {"question": content, "answer": answer, "userLogin": userLogin}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data


@app.get("/is-openapi-ok")
async def openapi():
    isOk = isinstance(OPENAI_API_KEY, str)
    return {"ok": isOk, }

@app.get("/ping")
async def ping():
    return "PONG"
