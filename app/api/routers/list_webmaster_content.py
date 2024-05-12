from fastapi import APIRouter, Depends
from app.auth.auth_resource_server import get_user_info
from app.mongo_models.webmaster_content_model import WebmasterContentListModel
from app.schemas.user_auth import User
from database.db import *

router = APIRouter()

@router.get(
    "/list_webmaster_content/",
    response_description="Get all webmaster content",
    response_model=WebmasterContentListModel,
    response_model_by_alias=False,
)
def list_proposed_content(user: User = Depends(get_user_info)):
    """
    Get all enriched content
    """
    return WebmasterContentListModel(webmaster_content_list=webmaster_content_collection.find())

