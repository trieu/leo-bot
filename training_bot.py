from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import os
import json

HOSTNAME = os.getenv("HOSTNAME")
FOLDER_RESOURCES = os.path.dirname(os.path.abspath(__file__)) + "/resources/"
FOLDER_TEMPLATES = FOLDER_RESOURCES + "templates"
DATA_FILE = "training_data.json"

app = FastAPI()

templates = Jinja2Templates(directory=FOLDER_TEMPLATES)


# Load existing data from the JSON file (if it exists)
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# Save data to the JSON file
def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
# Initialize the training data
training_data = load_data()

class TrainingDataItem(BaseModel):
    input: str
    output: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    data = {"request": request, "HOSTNAME": HOSTNAME, "mode": "support_bot"}
    return templates.TemplateResponse("motul_bot.html", data)

@app.get("/training_bot", response_class=HTMLResponse)
async def root(request: Request):
    data = {"request": request, "HOSTNAME": HOSTNAME, "mode": "training_bot"}
    return templates.TemplateResponse("motul_bot.html", data)


@app.post("/add_data")
async def add_data(input: str = Form(...), output: str = Form(...)):
    # Add the data to the list
    training_data.append("input: "+ input.strip())
    training_data.append("output: "+ output.strip())
    # Save the updated data to the JSON file
    save_data(training_data)
    
    return {"message": "Data added successfully", "current_data": training_data}

@app.get("/get_data", response_model=List[TrainingDataItem])
async def get_data():
    return training_data
