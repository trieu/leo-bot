import os


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from redis import Redis

from leoai.leo_chatbot import ask_question, translate_text, GOOGLE_GENAI_API_KEY
from leoai.leo_datamodel import Message
from leoai.leo_analytics import sentiment_pipe

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

@leobot.get("/is-ready")
async def is_leobot_ready():
    isReady = isinstance(GOOGLE_GENAI_API_KEY, str)
    return {"ok": isReady }


@leobot.get("/ping")
async def ping():
    return "PONG"


@leobot.get("/", response_class=HTMLResponse)
async def root(request: Request):
    data = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE" : LEOBOT_DEV_MODE}
    return templates.TemplateResponse("index.html", data)




# the main API of chatbot
@leobot.post("/ask")
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
        question_in_english = prompt

        if lang != "en":
            question_in_english = translate_text('en', prompt) 

        # translate if need
        context = " LEO CDP is LEO Customer Data Platform. "
        # context = context + " Today is " + date.today().strftime("%B %d, %Y") + ". "    
        answer = ask_question(context, format, lang, question_in_english)

        print("answer " + answer)
        data = {"question": question, "answer": answer, "userLogin": userLogin}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data


@leobot.post("/sentiment-analysis")
async def sentiment_analysis(msg: Message):
    prompt = msg.prompt
    print("sentiment_analysis msg "+prompt)
    userLogin = REDIS_CLIENT.hget(msg.usersession, 'userlogin')
    data = {"error": True}
    if userLogin == msg.userlogin:
        try:
            rs = sentiment_pipe(prompt)
            if len(rs) > 0 :
                data = rs[0]
        except Exception as error:
            print("An exception occurred:", error)
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data