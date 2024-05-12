from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime

class UserFeedbackTopicModel(BaseModel):
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    request_id: Optional[int] = Field(...)
    content_type: Optional[str] = Field(...)
    content_topic: Optional[str] = Field(...)
    url: Optional[str] = Field(...)
    data: Optional[dict] = Field(...) # these data are the full enc dict or faq dict or gen content or website content. `liked` is inside here
    last_update: Optional[datetime] = Field(default_factory=datetime.now)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "user_id": 0,
                "session_id": 0,
                "request_id": 0,
                "data": {"general":"dict"},
                "content_type": "string",
                "content_topic": "string",
            }
        }
    )
