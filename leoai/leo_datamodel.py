
from typing import Optional
from pydantic import BaseModel, Field

from leoai.leo_chatbot import TEMPERATURE_SCORE

# Data models
class Message(BaseModel):
    answer_in_language: Optional[str] = Field("en") # default is English
    answer_in_format: str = Field("html", description="the format of answer")
    context: str = Field("", description="the context of question")
    question: str = Field("", description="the question for Q&A ")
    temperature_score: float = Field(TEMPERATURE_SCORE, description="the temperature score of LLM ")
    prompt: str
    visitor_id: str