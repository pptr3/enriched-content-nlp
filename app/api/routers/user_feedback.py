from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime
from app.auth.auth_resource_server import get_user_info
from app.mongo_models.user_feedback_model import UserFeedbackTopicModel
from app.schemas.user_auth import User
from worker import update_embedding_feedback_task

router = APIRouter()

@router.post("/user_feedback", response_model=dict)
def user_feedback_endpoint(user_feedback: UserFeedbackTopicModel, user: User = Depends(get_user_info)):
    future = update_embedding_feedback_task.remote(user_feedback.model_dump())
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)