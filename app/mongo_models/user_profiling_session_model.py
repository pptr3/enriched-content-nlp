from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]

class UserProfilingSessionModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    user_embedding: Optional[list] = Field(...)
    topic_embedding_count: Optional[int] = Field(...)
    last_update: Optional[datetime] = Field(default_factory=datetime.now)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "user_id": 123,
                "session_id": 123,
                "user_embedding": [0.1, 0.2, 0.3],
                "topic_embedding_count": 5,
                "last_update": "2024-01-11T10:00:00"
            }
        },
    )
