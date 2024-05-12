from fastapi import APIRouter, Depends
from app.auth.auth_resource_server import get_user_info
from app.mongo_models.enriched_content_model import EnrichedContentListModel
from app.schemas.user_auth import User
from database.db import *

router = APIRouter()

@router.get(
    "/list_enriched_content/",
    response_description="Get all enriched content",
    response_model=EnrichedContentListModel,
    response_model_by_alias=False,
)
def list_proposed_content(user: User = Depends(get_user_info)):
    """
    Get all enriched content
    """
    return EnrichedContentListModel(enriched_content_list=enriched_content_collection.find())

