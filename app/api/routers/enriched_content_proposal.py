from fastapi import Depends, FastAPI, APIRouter
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
import ray
import time
from fastapi import FastAPI, APIRouter
import ray
import logging
import time
import os
import time
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from typing import Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from app.utils.data_extractor import DataExtractor
from app.utils.text_cleaner import TextCleaner
from app.utils.mqtt_publisher import MqttPublisher

from app.utils.formatter import Formatter
from app.api.routers.text_correction.text_correction import fix_text_with_corrige
from app.mongo_models.enriched_content_model import EnrichedContentModel
from app.mongo_models.linguistic_analysis_model import LinguisticAnalysisModel
from app.mongo_models.user_feedback_model import UserFeedbackTopicModel
from database.db import *
from config.config import get_global_config, get_local_config
from datetime import datetime
from pydantic import BaseModel
import fasttext
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from keybert import KeyBERT
import torch
import fasttext

import os
import shutil
import re
import requests
from openai import OpenAI
import torch
import json
import ast
from transformers import AutoModel
from huggingface_hub import HfApi, ModelFilter
from transformers import pipeline
import tempfile
import logging
from database.db import *
import ray
from fastapi import APIRouter
import time
import fasttext
from worker import enriched_content_proposal, nlp_service_text

router = APIRouter()

class EnrichedContentUrlRequest(BaseModel):
    url: Optional[str] = Field(...)
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    request_id: Optional[int] = Field(...)

class EnrichedContentTextRequest(BaseModel):
    text: Optional[str] = Field(...)
    user_id: Optional[int] = Field(...)
    session_id: Optional[int] = Field(...)
    request_id: Optional[int] = Field(...)

@router.post("/enriched_content_proposal")
async def enriched_content_proposal_func(request_data: EnrichedContentUrlRequest, user: User = Depends(get_user_info)):
    # Submit processing task to Ray
    future = enriched_content_proposal.remote(request_data.url,
                                              request_data.user_id,
                                              request_data.session_id,
                                              request_data.request_id)

    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)

@router.post("/nlp_service_text")
async def nlp_service_text_func(request_data: EnrichedContentTextRequest, user: User = Depends(get_user_info)):
    future = nlp_service_text.remote(request_data.text,
                                     request_data.user_id,
                                     request_data.session_id,
                                     request_data.request_id)

    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)