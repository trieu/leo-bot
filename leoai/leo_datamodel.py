
from typing import Optional
from pydantic import BaseModel, Field

# Data models
class Message(BaseModel):
    answer_in_language: Optional[str] = Field("en") # default is English
    answer_in_format: str = Field("html", description="the format of answer")
    context: str = Field("", description="the context of question")
    question: str
    prompt: str
    usersession: str
    userlogin: str