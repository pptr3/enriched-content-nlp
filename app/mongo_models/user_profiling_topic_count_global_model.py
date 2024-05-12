from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]

class UserProfilingTopicCountGlobalModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: Optional[int] = Field(...)
    user_history: Optional[list[dict]] = Field(...)
    last_update: Optional[datetime] = Field(default_factory=datetime.now)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "user_id": 1,
                "user_history":[{
                        "topic_name":"string",
                        "visit_occurrences": 2,
                        "last_visit":"timestamp"
                        },
                        {
                        "topic_name":"string",
                        "visit_occurrences": 3,
                        "last_visit":"timestamp"
                        }
                ],
                "last_update":"timestamp"
            }
        },
    )