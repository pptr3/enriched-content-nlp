from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
from worker import user_delete_task

router = APIRouter()

class UserDeletionModel(BaseModel):
    user_id: Optional[int] = Field(...)

@router.delete("/user-profile_reset") # TODO: chi chiama questo?
def user_delete_endpoint(user_deletion: UserDeletionModel, user: User = Depends(get_user_info)):
    user_delete_task(user_deletion.user_id)
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)