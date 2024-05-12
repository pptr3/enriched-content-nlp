from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
from config.config import get_global_config
from datetime import datetime
import json
from database.db import *
from worker import delete_model_task

router = APIRouter()

@router.put("/delete_nlp_model")
async def delete_nlp_model(user_model_name: str, user: User = Depends(get_user_info)):
    # Submit processing task to Ray
    future = delete_model_task.remote(user_model_name)
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)
