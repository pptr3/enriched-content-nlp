from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
from config.config import get_local_config
from app.mongo_models.local_config_model import LocalConfigModel
import json
from database.db import *

router = APIRouter()

@router.get(
    "/local_config/",
    response_description="Read the local configuration",
)
def read_local_config(user: User = Depends(get_user_info)):
    local_conf_data = get_local_config()    
    return JSONResponse(content=local_conf_data, status_code=200)

@router.put(
    "/local_config/",
    response_description="Update local config",
)
def update_local_config(new_local_config: LocalConfigModel = Body(...), user: User = Depends(get_user_info)):
    previous_local_config = get_local_config()
    previous_local_config_updated = get_local_config()
    previous_local_config_updated.update(new_local_config.model_dump(by_alias=True, exclude={"timestamp": True, "app_name": True, "CorrigeAlgorithm": True, "DataExtractor": True, "NLPAlgorithm": True, "FastText": True, "disk_free_space_percentile_threshold": True}))
    
    if previous_local_config != previous_local_config_updated: # it means that the new local config has been edited
        print("New version of local config", flush=True)
        store_local_config(LocalConfigModel(**previous_local_config_updated))
        with open("/app2/config/local_config.json", "w+") as fp:
            json.dump(previous_local_config_updated, fp)
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)