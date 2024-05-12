import os
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import Body
from bson import ObjectId

from app.mongo_models.enriched_content_model import EnrichedContentModel
from app.mongo_models.linguistic_analysis_model import LinguisticAnalysisModel, LinguisticAnalysisTextModel
from app.mongo_models.global_config_model import GlobalConfigModel
from app.mongo_models.local_config_model import LocalConfigModel
from app.mongo_models.webmaster_content_model import WebmasterContentModel
from app.mongo_models.user_profiling_global_model import UserProfilingGlobalModel
from app.mongo_models.user_profiling_session_model import UserProfilingSessionModel
from app.mongo_models.user_profiling_topic_count_global_model import UserProfilingTopicCountGlobalModel
from app.mongo_models.user_feedback_model import UserFeedbackTopicModel

# load environment variables from .env file
load_dotenv()

# retrieve MongoDB credentials from environment variables
mongo_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
mongo_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
mongo_url = f"mongodb://{mongo_username}:{mongo_password}@mongo-container:27017/admin"
client = MongoClient(mongo_url)

db = client["nlpservice"]  # here I put my custom db name
linguistic_analysis_collection = db.get_collection("linguistic_analysis") # here I put my custom db collection name
enriched_content_collection = db.get_collection("enriched_content") # here I put my custom db collection name
global_config_collection = db.get_collection("global_config")  # Collection for global_config
local_config_collection = db.get_collection("local_config")  # Collection for local_config
webmaster_content_collection = db.get_collection("webmaster_content")  # Collection for webmaster content
user_profiling_global_collection = db.get_collection("user_profiling_global")  # Collection for global user profiling
user_profiling_session_collection = db.get_collection("user_profiling_session") # Collection for session user profiling
user_profiling_topic_count_global_collection = db.get_collection("user_profiling_topic_count_global") # Collection for user profiling topic count global
user_feedback_collection = db.get_collection("user_feedback") #Collection for user feedback
linguistic_analysis_text_collection = db.get_collection("linguistic_analysis_text") #Collection for linguistic analysis text
enriched_content_text_collection = db.get_collection("enriched_content_text") #Collection for enriched content text


# TODO: uniform these functions as they are the same into three functions

def create_user_profiling_topic_count_global(user: UserProfilingTopicCountGlobalModel = Body(...)):
    """
    Insert a new user profiling topic count global.
    """
    new_user = user_profiling_topic_count_global_collection.insert_one(
        user.model_dump(by_alias=True, exclude=["id"])
    )
    created_user = user_profiling_topic_count_global_collection.find_one(
        {"_id": new_user.inserted_id}
    )
    return created_user

def update_user_profiling_topic_count_global(id: str, user: UserProfilingTopicCountGlobalModel = Body(...)):
    """
    Update individual fields of an existing user record.

    Only the provided fields will be updated.
    Any missing or `null` fields will be ignored.
    """
    
    user = {
        k: v for k, v in user.model_dump(by_alias=True).items() if v is not None
    }

    if len(user) >= 1:
        update_result = user_profiling_topic_count_global_collection.find_one_and_update(
            {"_id": ObjectId(id)},
            {"$set": user}
        )
        if update_result is not None:
            return update_result
    if (existing_user := user_profiling_topic_count_global_collection.find_one({"_id": id})) is not None:
        return existing_user

    raise None


def get_user_topic_count_global(user_id: int):
    """
    Get the record for a specific user topic count global, looked up by `user_id`.
    """
    if (
        user := user_profiling_topic_count_global_collection.find_one({"user_id": user_id})
    ) is not None:
        return user

    return None # user_id not present


def create_user_profiling_global(user: UserProfilingGlobalModel = Body(...)):
    """
    Insert a new user profiling global.
    """
    new_user = user_profiling_global_collection.insert_one(
        user.model_dump(by_alias=True, exclude=["id"])
    )
    created_user = user_profiling_global_collection.find_one(
        {"_id": new_user.inserted_id}
    )
    return created_user

def update_user_profiling_global(id: str, user: UserProfilingGlobalModel = Body(...)):
    """
    Update individual fields of an existing user record.

    Only the provided fields will be updated.
    Any missing or `null` fields will be ignored.
    """
    
    user = {
        k: v for k, v in user.model_dump(by_alias=True).items() if v is not None
    }

    if len(user) >= 1:
        update_result = user_profiling_global_collection.find_one_and_update(
            {"_id": ObjectId(id)},
            {"$set": user}
        )
        if update_result is not None:
            return update_result
    if (existing_user := user_profiling_global_collection.find_one({"_id": id})) is not None:
        return existing_user

    raise None



def get_user_global(user_id: int):
    """
    Get the record for a specific user global, looked up by `user_id`.
    """
    if (
        user := user_profiling_global_collection.find_one({"user_id": user_id})
    ) is not None:
        return user

    return None # user_id not present


def create_user_profiling_session(user: UserProfilingSessionModel = Body(...)):
    """
    Insert a new user profiling global.

    A unique `id` will be created and provided in the response.
    """
    new_user = user_profiling_session_collection.insert_one(
        user.model_dump(by_alias=True, exclude=["id"])
    )
    created_user = user_profiling_session_collection.find_one(
        {"_id": new_user.inserted_id}
    )
    return created_user


def update_user_profiling_session(id: str, user: UserProfilingSessionModel = Body(...)):
    """
    Update individual fields of an existing user record.

    Only the provided fields will be updated.
    Any missing or `null` fields will be ignored.
    """
    
    user = {
        k: v for k, v in user.model_dump(by_alias=True).items() if v is not None
    }

    if len(user) >= 1:
        update_result = user_profiling_session_collection.find_one_and_update(
            {"_id": ObjectId(id)},
            {"$set": user}
        )
        if update_result is not None:
            return update_result
    if (existing_user := user_profiling_session_collection.find_one({"_id": id})) is not None:
        return existing_user

    raise None

def get_user_session(user_id: int, session_id: int):
    """
    Get the record for a specific user and session, looked up by 'user_id' and 'session_id'.
    """
    if (user := user_profiling_session_collection.find_one({"user_id": user_id, "session_id": session_id})) is not None:
        return user
    return None # user_id not present




def store_linguistic_analysis(linguistic_analysis: LinguisticAnalysisModel = Body(...)):
    saved_linguistic_analysis = linguistic_analysis_collection.insert_one(
        linguistic_analysis.model_dump(by_alias=True, exclude=["id"])
    )
    
    created_linguistic_analysis = linguistic_analysis_collection.find_one(
        {"_id": saved_linguistic_analysis.inserted_id}
    )
    return created_linguistic_analysis


def store_enriched_content(enriched_content: EnrichedContentModel = Body(...)):
    saved_enriched_content = enriched_content_collection.insert_one(
        enriched_content.model_dump(by_alias=True, exclude=["id"])
    )
    
    created_enriched_content = enriched_content_collection.find_one(
        {"_id": saved_enriched_content.inserted_id}
    )
    return created_enriched_content

def store_global_config(new_global_config: GlobalConfigModel = Body(...)):
    saved_global_config = global_config_collection .insert_one(
        new_global_config.model_dump(by_alias=True, exclude=["id"])
    )

    created_global_config = global_config_collection.find_one(
        {"_id": saved_global_config.inserted_id}
    )
    return created_global_config

def store_local_config(local_config: LocalConfigModel = Body(...)):

    saved_local_config = local_config_collection.insert_one(
        local_config.model_dump(by_alias=True, exclude=["id"])
    )

    created_local_config = local_config_collection.find_one(
        {"_id": saved_local_config.inserted_id}
    )
    return created_local_config


def store_webmaster_content(webmaster_content: WebmasterContentModel = Body(...)):
    saved_webmaster_content = webmaster_content_collection.insert_one(
        webmaster_content.model_dump(by_alias=True, exclude=["id"])
    )

    created_webmaster_content = webmaster_content_collection.find_one(
        {"_id": saved_webmaster_content.inserted_id}
    )
    return created_webmaster_content

def store_user_feedback(new_user_feedback: UserFeedbackTopicModel = Body(...)):
    saved_user_feedback = user_feedback_collection.insert_one(
        new_user_feedback.model_dump(by_alias=True, exclude=["id"])
    )

    created_user_feedback = user_feedback_collection.find_one(
        {"_id": saved_user_feedback.inserted_id}
    )
    return created_user_feedback


def store_linguistic_analysis_text(linguistic_analysis_text: LinguisticAnalysisTextModel = Body(...)):
    saved_linguistic_analysis_text = linguistic_analysis_text_collection.insert_one(
        linguistic_analysis_text.model_dump(by_alias=True, exclude=["id"])
    )
    
    created_linguistic_analysis_text = linguistic_analysis_text_collection.find_one(
        {"_id": saved_linguistic_analysis_text.inserted_id}
    )
    return created_linguistic_analysis_text

def store_enriched_content_text(enriched_content_text: EnrichedContentModel = Body(...)):
    saved_enriched_content_text = enriched_content_text_collection.insert_one(
        enriched_content_text.model_dump(by_alias=True, exclude=["id"])
    )
    
    created_enriched_content_text = enriched_content_text_collection.find_one(
        {"_id": saved_enriched_content_text.inserted_id}
    )
    return created_enriched_content_text