import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from redis import Redis
from pathlib import Path
import json

from leoai.ai_chatbot import ask_question, GEMINI_API_KEY, translate_text, detect_language, extract_data_from_chat_message_by_ai
from leoai.leo_datamodel import Message, UpdateProfileEvent, ChatMessage, TrackedEvent

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
    isReady = isinstance(GEMINI_API_KEY, str)
    return {"ok": isReady}


@leobot.get("/ping", response_class=PlainTextResponse)
async def ping():
    return "PONG"


@leobot.post("/event-game/put", response_class=JSONResponse)
async def event_tracking(e: TrackedEvent):
    # Use .model_dump(mode="json") to handle datetime serialization
    event_data_dict = e.model_dump(mode="json")

    # Convert the dictionary to a JSON string
    event_data_json = json.dumps(event_data_dict, indent=4, ensure_ascii=False)

    # Define the JSON file path
    file_path = Path("tracked-event.json")

    # Save the JSON string to a file, TODO save into AWS S3
    with file_path.open("w", encoding="utf-8") as file:
        file.write(event_data_json)
    
    return {"message": "Data successfully sent to Firehose", "recordId": "1"}


@leobot.get("/", response_class=HTMLResponse)
async def root(request: Request):
    ts = int(time.time())
    data = {"request": request, "HOSTNAME": HOSTNAME, "LEOBOT_DEV_MODE": LEOBOT_DEV_MODE, 'timestamp': ts}
    return templates.TemplateResponse("index.html", data)


@leobot.get("/get-visitor-info", response_class=JSONResponse)
async def get_visitor_info(visitor_id: str):
    isReady = isinstance(GEMINI_API_KEY, str)
    if not isReady:        
        return {"answer": "GEMINI_API_KEY is empty", "error_code": 501}
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
    answer_in_language = msg.answer_in_language
    context = msg.context
       
    if len(question) > 1000 :
        return {"answer": "Question must be less than 1000 characters!", "error": True, "error_code": 510}

    print("context: "+context)
    print("question: "+question)
    print("visitor_id: " + visitor_id)
    print("profile_id: "+profile_id)

    if leobot_ready:        
                    
        format = msg.answer_in_format
        temperature_score = msg.temperature_score
            
        # translate if need
        answer = ask_question(context=context, question=question,temperature_score=temperature_score,
                              answer_in_format=format, target_language=answer_in_language)
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
    event_id = e.event_id
    # get profile features from ArangoDB by profile_id 
    # LINK: https://g.co/gemini/share/b34bf889420b
    data = {"error": True}
    if len(profile_id) > 0 :
        context = "You are a profile analysis system."
    else:
        data = {"answer": "Invalid usersession", "error": True}
    return data

@leobot.post("/extract-data-from-chat-event", response_class=JSONResponse)
async def extract_data_from_chat_event(msg: ChatMessage):
    # Extract data from the chat message
    extracted_data = extract_data_from_chat_message_by_ai(msg)

    # Print the pretty-printed JSON string
    # json_str = json.dumps(extracted_data, indent=4, ensure_ascii=False)
    # print(json_str)
    return extracted_data