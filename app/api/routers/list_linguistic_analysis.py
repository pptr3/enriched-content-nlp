from fastapi import APIRouter, Depends
from app.auth.auth_resource_server import get_user_info
from app.mongo_models.linguistic_analysis_model import LinguisticAnalysisListModel
from app.schemas.user_auth import User
from database.db import *

router = APIRouter()

@router.get(
    "/list_linguistic_analysis/",
    response_description="Get all linguistic analysis",
    response_model=LinguisticAnalysisListModel,
    response_model_by_alias=False,
)
def list_proposed_content(user: User = Depends(get_user_info)):
    """
    Get all linguistic analysis 
    """
    return LinguisticAnalysisListModel(linguistic_analysis_list=linguistic_analysis_collection.find())
