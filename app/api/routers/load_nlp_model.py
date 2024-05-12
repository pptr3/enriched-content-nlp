import re
import shutil
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
from config.config import get_local_config, get_global_config
from database.db import *

from pydantic import BaseModel, Field
from huggingface_hub import HfApi
from typing import Optional
from pydantic import BaseModel, Field

from worker import load_model_task

router = APIRouter()

class LoadModel(BaseModel):
    hf_url: Optional[str] = Field(...)
    tag: Optional[str] = Field(...) 

@router.put(
    "/load_nlp_model/",
    response_description="""Load NLP model from HuggingFace, choose a key in this dictionary as tag: {
        "Generative": "text-generation",
        "FAQ": "text-generation",
        "NER": "token-classification",
        "Keyword extraction": "summarization",
        "Topic extraction": "zero-shot-classification"}"""
)
def load_nlp_model(request_data: LoadModel, user: User = Depends(get_user_info)):
    def get_total_and_available_disk_space(path='/'):
        disk_info = shutil.disk_usage(path)
        space_total_gb = disk_info.total / (1024 ** 3)
        space_available_gb = disk_info.free / (1024 ** 3)
        return space_total_gb, space_available_gb
    
    tags = {
        "Generative": "text-generation",
        "FAQ": "text-generation",
        "NER": "token-classification",
        "Keyword extraction": "summarization",
        "Topic extraction": "zero-shot-classification"
    }
    model_tag = tags[request_data.tag]
    # extract the model name from the URL
    match = re.search(r"huggingface.co/([^/]+/[^/]+)", request_data.hf_url)
    if not match:
        return JSONResponse(content={"status": 422}, status_code=422)

    user_model_name = match.group(1)
    api = HfApi()
    try:
        modelinfo = api.model_info(user_model_name, files_metadata=True)
    except Exception as e:
        print(e, flush=True)
        return JSONResponse(content={"status": 422}, status_code=422)        
    if model_tag not in modelinfo.pipeline_tag:
        return JSONResponse(content={"status": 422}, status_code=422)
    modelsize_gb = sum([sibling.size for sibling in modelinfo.siblings]) / (1024 ** 3)
    total_disk_size_gb , available_disk_size_gb = get_total_and_available_disk_space('/')
    if (available_disk_size_gb - modelsize_gb) / total_disk_size_gb <= get_local_config()["disk_free_space_percentile_threshold"]:
        return JSONResponse(content={"status": 422}, status_code=422)

    future = load_model_task.remote(user_model_name)
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)
