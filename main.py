from langchain.llms import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os

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


@app.get("/")
async def root():
    return {"SERVICE_NAME": SERVICE_NAME}


@app.get("/ask")
async def ask(question: str):
    print(question)
    answer = llm(question)
    return {"question": question, "answer": answer}


@app.get("/is-openapi-ok")
async def openapi():
    isOk = isinstance(OPENAI_API_KEY, str)
    return {"ok": isOk, }
