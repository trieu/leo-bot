
from typing import Optional
from pydantic import BaseModel, Field

DEFAULT_TEMPERATURE_SCORE = 1.0

# Data models
class Message(BaseModel):
    answer_in_language: Optional[str] = Field("en") # default is English
    answer_in_format: str = Field("html", description="the format of answer")
    context: str = Field("You are a creative chatbot.", description="the context of question")
    question: str = Field("", description="the question for Q&A ")
    temperature_score: float = Field(DEFAULT_TEMPERATURE_SCORE, description="the temperature score of LLM ")
    prompt: str
    visitor_id: str = Field("", description="the visitor id ")
    
# Data models
class UpdateProfileEvent(BaseModel):
    profile_id: str = Field("", description="the ID of CDP profile")
    event_id: str = Field("", description="the ID of tracking event")
    asset_group_id: str = Field("", description="the ID of Digital Asset Group")
    asset_type: int = Field("", description="the type of Digital Asset")
    
# Data models
class ChatMessage(BaseModel):
    profile_id: str = Field("", description="the ID of CDP profile")
    event_id: str = Field("", description="the ID of tracking event")
    content: str = Field("", description="the content of chat message")