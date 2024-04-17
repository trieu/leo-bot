import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from redis import Redis

from leoai.leo_chatbot import ask_question, translate_text, detect_language, GOOGLE_GENAI_API_KEY
from leoai.leo_datamodel import Message
from leoai.leo_datamodel import UpdateProfileEvent

load_dotenv(override=True)

VERSION = "1.0.0"
SERVICE_NAME = "LEO BOT VERSION:" + VERSION

LEOBOT_DEV_MODE = os.getenv("LEOBOT_DEV_MODE") == "true"
HOSTNAME = os.getenv("HOSTNAME")
REDIS_USER_SESSION_HOST = os.getenv("REDIS_USER_SESSION_HOST")
REDIS_USER_SESSION_PORT = os.getenv("REDIS_USER_SESSION_PORT")

print("HOSTNAME " + HOSTNAME)
print("LEOBOT_DEV_MODE " + str(LEOBOT_DEV_MODE))

# Redis Client to get User Session
REDIS_CLIENT = Redis(host=REDIS_USER_SESSION_HOST,  port=REDIS_USER_SESSION_PORT, decode_responses=True)
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
leobot.mount("/resources", StaticFiles(directory=FOLDER_RESOURCES), name="resources")
templates = Jinja2Templates(directory=FOLDER_TEMPLATES)

def is_visitor_ready(visitor_id:str):
    return REDIS_CLIENT.hget(visitor_id, 'chatbot') == "leobot" or LEOBOT_DEV_MODE

##### API handlers #####


@leobot.get("/is-ready", response_class=JSONResponse)
@leobot.post("/is-ready", response_class=JSONResponse)
async def is_leobot_ready():
    isReady = isinstance(GOOGLE_GENAI_API_KEY, str)
    return {"ok": isReady}


@leobot.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@leobot.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, 'timestamp': ts}
    return templates.TemplateResponse("index.html", data)


@leobot.get("/get-visitor-info", response_class=JSONResponse)
async def get_visitor_info(visitor_id: str):
    isReady = isinstance(GOOGLE_GENAI_API_KEY, str)
    if not isReady:        
        return {"answer": "GOOGLE_GENAI_API_KEY is empty", "error_code": 501}
    if len(visitor_id) == 0: 
        return {"answer": "visitor_id is empty ", "error": True, "error_code": 500}
    profile_id = REDIS_CLIENT.hget(visitor_id, 'profile_id')
    if profile_id is None or len(profile_id) == 0: 
        if LEOBOT_DEV_MODE : 
            return {"answer": "local_dev", "error_code": 0}
        else:
            return {"answer": "Not found any profile in CDP", "error": True, "error_code": 404}
    name = str(REDIS_CLIENT.hget(visitor_id, 'name'))
    return {"answer": name, "error_code": 0}


# the main API of chatbot
@leobot.post("/ask", response_class=JSONResponse)
async def ask(msg: Message):
    visitor_id = msg.visitor_id
    if len(visitor_id) == 0: 
        return {"answer": "visitor_id is empty ", "error": True, "error_code": 500}
    
    if LEOBOT_DEV_MODE:
        profile_id = "0"
    else:
        profile_id = REDIS_CLIENT.hget(visitor_id, 'profile_id')
        if profile_id is None or len(profile_id) == 0: 
            return {"answer": "Not found any profile in CDP", "error": True, "error_code": 404}
    
    leobot_ready = is_visitor_ready(visitor_id)
    question = msg.question
    prompt = msg.prompt
    lang_of_question = msg.answer_in_language
    
    if len(question) > 1000 or len(prompt) > 1000 :
        return {"answer": "Question must be less than 1000 characters!", "error": True, "error_code": 510}

    print("question: "+question)
    print("prompt: "+prompt)
    print("visitor_id: " + visitor_id)
    print("profile_id: "+profile_id)

    if leobot_ready:        
        if lang_of_question == "" :
            lang_of_question = detect_language(question)        
             
        format = msg.answer_in_format
        temperature_score = msg.temperature_score
        question_in_english = prompt

        if lang_of_question != "en":
            # our model can only understand English        
            question_in_english = translate_text(prompt, 'en')
            
        # translate if need
        context = "CDP is Customer Data Platform."
        # context = context + " Today is " + date.today().strftime("%B %d, %Y") + ". "
        answer = ask_question(context, format, lang_of_question, question_in_english, temperature_score)
        print("answer " + answer)
        data = {"question": question,
                "answer": answer, "visitor_id": visitor_id, "error_code": 0}
    else:
        data = {"answer": "Your profile is banned due to Violation of Terms", "error": True, "error_code": 666}
    return data


@leobot.post("/sentiment-analysis", response_class=JSONResponse)
async def sentiment_analysis(msg: Message):
    feedback_text = msg.prompt
    userLogin = REDIS_CLIENT.hget(msg.usersession, 'userlogin')
    data = {"error": True}
    if userLogin == msg.userlogin:
        context = "You are a sentiment analysis system."
        translated_feedback = translate_text(feedback_text, 'en')
        # print("sentiment_analysis translated_feedback \n "+translated_feedback)
        sentiment_command = 'Give rating score from 1 to 100 if this text is positive customer feedback: '
        prompt = sentiment_command + translated_feedback
        answer = ask_question(context, "text", "en", prompt, 1)
        data = {"answer": int(answer)}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data

@leobot.post("/profile-analysis", response_class=JSONResponse)
async def profile_analysis(e: UpdateProfileEvent):
    profile_id = e.profile_id
    # get profile features from ArangoDB by profile_id 
    # LINK: https://g.co/gemini/share/b34bf889420b
    data = {"error": True}
    if len(profile_id) > 0 :
        context = "You are a sentiment analysis system."
        translated_feedback = translate_text(feedback_text, 'en')
        # print("sentiment_analysis translated_feedback \n "+translated_feedback)
        sentiment_command = 'Give rating score from 1 to 100 if this text is positive customer feedback: '
        prompt = sentiment_command + translated_feedback
        answer = ask_question(context, "text", "en", prompt, 1)
        data = {"answer": int(answer)}
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data