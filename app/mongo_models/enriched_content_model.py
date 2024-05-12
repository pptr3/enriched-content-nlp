from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]


class EnrichedContentModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    request_id: Optional[int] = Field(...)
    elapsed_time: Optional[float] = Field(...)
    encyclopedic_content: Optional[dict] = Field(...)
    generative_content: Optional[str] = Field(...)
    website_content: Optional[dict] = Field(...)
    faq: Optional[dict] = Field(...)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "user_id": 1,
                "session_id": 1,
                "request_id": 1,
                "elapsed_time": 20.5,
                "encyclopedic_content": {'wiki_text': 'text', 'wiki_url': 'url', 'title_page': 'title', 'page_description': 'description', 'page_keywords': ["keyword"]},
                "generative_content": "test generativo",
                "website_content": {"topic1": "link1", "topic2": "link2"},
                "faq": {"content": [{"question": "question", "answer": "answer"}], "topic": "topic"},
                "timestamp": "2024-01-11T10:00:00"
            }
        },
    )


class EnrichedContentListModel(BaseModel):
    """
    A container holding a list of `EnrichedContentModel` instances.
    """
    enriched_content_list: List[EnrichedContentModel]
