import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from redis import Redis

from leoai.leo_chatbot import ask_question, translate_text, GOOGLE_GENAI_API_KEY
from leoai.leo_datamodel import Message

VERSION = "1.0.0"
SERVICE_NAME = "LEO BOT VERSION:" + VERSION

LEOBOT_DEV_MODE = os.getenv("LEOBOT_DEV_MODE") == "true"
HOSTNAME = os.getenv("HOSTNAME")
REDIS_USER_SESSION_HOST = os.getenv("REDIS_USER_SESSION_HOST")
REDIS_USER_SESSION_PORT = os.getenv("REDIS_USER_SESSION_PORT")

# Redis Client to get User Session
REDIS_CLIENT = Redis(host=REDIS_USER_SESSION_HOST, port=REDIS_USER_SESSION_PORT, decode_responses=True)
FOLDER_RESOURCES = os.path.dirname(os.path.abspath(__file__)) + "/resources/"
FOLDER_TEMPLATES = FOLDER_RESOURCES + "templates"

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
leobot.mount("/resources", StaticFiles(directory = FOLDER_RESOURCES), name="resources")
templates = Jinja2Templates(directory= FOLDER_TEMPLATES )

##### API handlers #####

@leobot.get("/is-ready", response_class=JSONResponse)
@leobot.post("/is-ready", response_class=JSONResponse)
async def is_leobot_ready():
    isReady = isinstance(GOOGLE_GENAI_API_KEY, str)
    return {"ok": isReady }


@leobot.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@leobot.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE" : LEOBOT_DEV_MODE, 'timestamp':ts}
    return templates.TemplateResponse("index.html", data)


# the main API of chatbot
@leobot.post("/ask", response_class=JSONResponse)
async def ask(msg: Message):
    userLogin = REDIS_CLIENT.hget(msg.usersession, 'userlogin')    
    question = msg.question
    prompt = msg.prompt
    
    print("question: "+question)
    print("prompt: "+prompt)
    print("question.userlogin: "+msg.userlogin)
    print("usersession: "+ msg.usersession)    
    print("userLogin: "+userLogin)
    
    if userLogin == msg.userlogin:
        # answer = llm(question)

        # our model can only understand English
        lang = msg.answer_in_language
        format = msg.answer_in_format
        temperature_score = msg.temperature_score
        question_in_english = prompt

        if lang != "en":
            question_in_english = translate_text(prompt, 'en') 

        # translate if need
        context = " LEO CDP is LEO Customer Data Platform. "
        # context = context + " Today is " + date.today().strftime("%B %d, %Y") + ". "    
        answer = ask_question(context, format, lang, question_in_english, temperature_score)

        print("answer " + answer)
        data = {"question": question, "answer": answer, "userLogin": userLogin}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data


@leobot.post("/sentiment-analysis", response_class=JSONResponse)
async def sentiment_analysis(msg: Message):
    feedback_text = msg.prompt
    
    userLogin = REDIS_CLIENT.hget(msg.usersession, 'userlogin')
    data = {"error": True}
    if userLogin == msg.userlogin:
        context = "You are the sentiment analysis system."
        translated_feedback = translate_text(feedback_text, 'en') 
        print("sentiment_analysis translated_feedback \n "+translated_feedback)
        sentiment_command = 'Give rating score from 1 to 100 if this text is positive customer feedback: ';
        prompt = sentiment_command + translated_feedback
        answer = ask_question(context, "text", "en", prompt, temperature_score = 1)
        data = {"answer": int(answer)}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data