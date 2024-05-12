from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Optional, List
from pydantic import ConfigDict, BaseModel, Field
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]


class WebmasterContentModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    webpage_url: Optional[str] = Field(...)
    encyclopedic_content: Optional[str] = Field(...)
    generative_content: Optional[str] = Field(...)
    website_content: Optional[list] = Field(...)
    faqs: Optional[list] = Field(...)
    keywords: Optional[list] = Field(...)
    topics: Optional[list] = Field(...)
    ners: Optional[list] = Field(...)
    corrige_corrections: Optional[list] = Field(...)
    webpage_text: Optional[str] = Field(...)
    word_count: Optional[int] = Field(...)
    title_page: Optional[str] = Field(...)
    page_description: Optional[str] = Field(...)
    page_keywords: Optional[list[str]] = Field(...)
    page_author: Optional[str] = Field(...)
    page_lang: Optional[str] = Field(...)
    

class WebmasterContentListModel(BaseModel):
    """
    A container holding a list of `WebmasterContentModel` instances.
    """
    webmaster_content_list: List[WebmasterContentModel]
