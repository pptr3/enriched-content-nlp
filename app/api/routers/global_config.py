from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.auth.auth_resource_server import get_user_info
from app.schemas.user_auth import User
from config.config import get_global_config
from app.mongo_models.global_config_model import GlobalConfigModel
import json
from database.db import *

router = APIRouter()

@router.get(
    "/config_algorithms/",
    response_description="Return the algorithms configuration",
)
def read_config_algorithms(user: User = Depends(get_user_info)):
    config_algorithms = get_global_config()["models"]
    return JSONResponse(content=config_algorithms, status_code=200)

@router.get(
    "/config_features/",
    response_description="Return the features configuration",
)
def read_config_features(user: User = Depends(get_user_info)):
    config_features = get_global_config()
    del config_features["models"] # we delete the algorithms configurations and keep just the features config
    return JSONResponse(content=config_features, status_code=200)

@router.put(
    "/config_features/",
    response_description="Update config_features",
)
def update_global_config(new_global_config: GlobalConfigModel = Body(...), user: User = Depends(get_user_info)):
    previous_global_config = get_global_config()
    previous_global_config_updated = get_global_config()
    previous_global_config_updated.update(new_global_config.model_dump(by_alias=True, exclude={"timestamp": True, "models": True, "instance_type": True}))
    
    if previous_global_config != previous_global_config_updated: # it means that the new global config has been edited
        print("New version of global config", flush=True)
        store_global_config(GlobalConfigModel(**previous_global_config_updated))
        with open("/app2/config/global_config.json", "w+") as fp:
            json.dump(previous_global_config_updated, fp)
    response_content = {"status": 200}
    return JSONResponse(content=response_content, status_code=200)




