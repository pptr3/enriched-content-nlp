from pymongo import MongoClient
from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import ConfigDict
from pydantic.functional_validators import BeforeValidator
from fastapi import Body
from typing_extensions import Annotated
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]


class LinguisticAnalysisModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    url: Optional[str] = Field(...)
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    request_id: Optional[int] = Field(...)
    elapsed_time: Optional[float] = Field(...)
    keywords: Optional[list] = Field(...)
    faq: Optional[dict] = Field(...)
    topic: Optional[list] = Field(...)
    ner: Optional[list] = Field(...)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    progressive_timings: Optional[dict] = Field(...)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "url": "url",
                "user_id": 1,
                "session_id": 1,
                "request_id": 1,
                "elapsed_time": 20.5,
                "keywords": [["k1", 0.9], ["k2", 0.8]],
                "faq": {"content": [{"question": "question", "answer": "answer"}], "topic": "topic"},
                "topic": ["topic", 0.7],
                "ner": [{"entity_group":"LOC","score": 0.99, "word": "Santa Maria del Monte"}, {"entity_group":"LOC", "score": 0.99, "word": "Santa Maria del Monte"}],
                "timestamp": "2024-01-11T10:00:00",
                "progressive_timings": {}
            }
        },
    )


class LinguisticAnalysisListModel(BaseModel):
    """
    A container holding a list of `LinguisticAnalysisModel` instances.
    """

    linguistic_analysis_list: List[LinguisticAnalysisModel]



class LinguisticAnalysisTextModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    text: Optional[str] = Field(...)
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    request_id: Optional[int] = Field(...)
    elapsed_time: Optional[float] = Field(...)
    keywords: Optional[list] = Field(...)
    faq: Optional[dict] = Field(...)
    topic: Optional[list] = Field(...)
    ner: Optional[list] = Field(...)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    progressive_timings: Optional[dict] = Field(...)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "text": "text",
                "user_id": 1,
                "session_id": 1,
                "request_id": 1,
                "elapsed_time": 20.5,
                "keywords": [["k1", 0.9], ["k2", 0.8]],
                "faq": {"content": [{"question": "question", "answer": "answer"}], "topic": "topic"},
                "topic": ["topic", 0.7],
                "ner": [{"entity_group":"LOC","score": 0.99, "word": "Santa Maria del Monte"}, {"entity_group":"LOC", "score": 0.99, "word": "Santa Maria del Monte"}],
                "timestamp": "2024-01-11T10:00:00",
                "progressive_timings": {}
            }
        },
    )


class LinguisticAnalysisTextListModel(BaseModel):
    """
    A container holding a list of `LinguisticAnalysisTextModel` instances.
    """

    linguistic_analysis_text_list: List[LinguisticAnalysisTextModel]

