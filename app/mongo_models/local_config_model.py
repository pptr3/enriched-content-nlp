from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated, Optional, Dict, Any
from datetime import datetime
from config.config import get_local_config


class LocalConfigModel(BaseModel):
    app_name: Optional[str] = Field(default=get_local_config()["app_name"])
    CorrigeAlgorithm: Optional[dict] = Field(default=get_local_config()["CorrigeAlgorithm"])
    DataExtractor: Optional[dict] = Field(default=get_local_config()["DataExtractor"])
    NLPAlgorithm: Optional[dict] = Field(default=get_local_config()["NLPAlgorithm"])
    FastText: Optional[dict] = Field(default=get_local_config()["FastText"])
    disk_free_space_percentile_threshold: Optional[float] = Field(default=get_local_config()["disk_free_space_percentile_threshold"])
    
    publisher_ip: Optional[str] = Field()
    publisher_port_kpi: Optional[int] = Field()
    publisher_port_enriched_content: Optional[int] = Field()
    Corrige_username: Optional[str] = Field(...)
    Corrige_password: Optional[str] = Field(...)
    Corrige_id_utente: Optional[str] = Field(...)
    openai_api_key: Optional[str] = Field(...)
    google_search_api_key: Optional[str] = Field(...)
    NLPAlgorithm_ner_th: Optional[float] = Field(...)
    keyword_instead_topic: Optional[bool] = Field(...)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra= {
            "example": {
                "publisher_ip": get_local_config()["publisher_ip"],
                "publisher_port_kpi": get_local_config()["publisher_port_kpi"],
                "publisher_port_enriched_content": get_local_config()["publisher_port_enriched_content"],
                "Corrige_id_utente": get_local_config()["Corrige_id_utente"],
                "NLPAlgorithm_ner_th": get_local_config()["NLPAlgorithm_ner_th"],
                "Corrige_username": get_local_config()["Corrige_username"],
                "Corrige_password": get_local_config()["Corrige_password"],
                "openai_api_key": get_local_config()["openai_api_key"],
                "google_search_api_key": get_local_config()["google_search_api_key"],
                "keyword_instead_topic": get_local_config()["keyword_instead_topic"]
            }
        }
    )

