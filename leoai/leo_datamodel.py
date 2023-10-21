
from typing import Optional
from pydantic import BaseModel, Field

# Data models
class Message(BaseModel):
    answer_in_language: Optional[str] = Field("en") # default is English
    content: str
    prompt: Optional[str] = Field(None, description="the question")
    usersession: str
    userlogin: str
