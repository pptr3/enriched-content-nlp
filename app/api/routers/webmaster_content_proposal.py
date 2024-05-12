from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
from worker import webmaster_content_proposal

router = APIRouter()

class WebmasterRequest(BaseModel):
    webpage_url: Optional[str] = Field(...)

@router.post("/webmaster_content_proposal") # TODO: mettere questo endpoint al sicuro e capire chi mi chiama per cambiare il tipo di dato che mi ariva
def process_webmaster(request_data: WebmasterRequest, user: User = Depends(get_user_info)):
    future = webmaster_content_proposal.remote(request_data.webpage_url)
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)

