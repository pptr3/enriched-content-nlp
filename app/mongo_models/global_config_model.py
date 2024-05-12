from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime
from config.config import get_global_config


# fields starting with underscore are not updatable. The same for `timestamp` because it has a `default_factory`
class GlobalConfigModel(BaseModel):
    models: Optional[dict] = Field(default=get_global_config()["models"])
    instance_type: Optional[str] = Field(default=get_global_config()["instance_type"])
    
    link_number: Optional[int] = Field(...)
    FAQ_number: Optional[int] = Field(...)
    encyclopedic_content: Optional[bool] = Field(...)
    generated_content: Optional[bool] = Field(...)
    FAQ_content: Optional[bool] = Field(...)
    websites_content: Optional[bool] = Field(...)
    reinforcement_mechanism: Optional[bool] = Field(...)
    coherence_evaluation: Optional[bool] = Field(...)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "link_number": 3,
                "FAQ_number": 3,
                "encyclopedic_content": True,
                "generated_content": True,
                "FAQ_content": True,
                "websites_content": True,
                "reinforcement_mechanism": True,
                "coherence_evaluation": True
            }
        }
    )
