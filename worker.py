from fastapi import FastAPI, APIRouter
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
from app.mongo_models.linguistic_analysis_model import LinguisticAnalysisModel, LinguisticAnalysisTextModel
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

from app.utils.nlp_algorithm import UnsharedModelsActor, SharedModelsActor

import ray
from fastapi import APIRouter
import time
import fasttext


fasttext_model = get_local_config()["FastText"]["path_reduced_model"]
gensim_fasttext_model = get_local_config()["FastText"]["path_reduced_model"] # NOTE: this is the same model as `fasttext_model`
ner_tokenizer = f"app/models/{get_global_config()['models']['ner_algorithm']['model_name']}"
ner_model = f"app/models/{get_global_config()['models']['ner_algorithm']['model_name']}" # NOTE: this is the same model as `ner_tokenizer`
        
shared_models = SharedModelsActor.remote(fasttext_model, gensim_fasttext_model, ner_model, ner_tokenizer)

@ray.remote
def nlp_service_text(text:str, user_id:int, session_id:int, request_id:int):
    try:
        logging.basicConfig(filename="./logs/logs_nlp_service_text.txt",
                            format='pid(%(process)d) %(asctime)s %(levelname)-8s %(funcName)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info(f"input text: {text}")
        kw_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"
        topic_classifier = f"app/models/{get_global_config()['models']['topic_extraction_algorithm']['model_name']}"
          
        nlp_algorithm = UnsharedModelsActor.remote(get_local_config(), get_global_config(), kw_model, topic_classifier)

        requested_at = datetime.now()
        progressive_timings = {}
        start_time = time.time()
        global_config = get_global_config()
        local_config = get_local_config()
        text_cleaner = TextCleaner(global_config)
        data_extractor = DataExtractor(local_config)
        mqtt_publisher = MqttPublisher(local_config)

        empty_encyclopedic_content = {'wiki_text': '', 'wiki_url': '', 'title_page': '', 'page_description': '', 'page_keywords': []}

        all_metadata = {"page_description": None, "title_page": None, "page_keywords": None, "page_author": None, "page_lang": None}
        text_and_metadata = {'text': text, 'all_metadata': all_metadata}
        html_metadata_keywords = text_and_metadata["all_metadata"]["page_keywords"]
        if text is None:
            logger.warning("text is None so publish_empty is called")
            return mqtt_publisher.publish_empty(user_id, session_id, request_id, text, empty_encyclopedic_content)

        elapsed_time_data_extraction = 0 # NOTE: there is no data extraction

        start_time_text_processing = time.time()
        cleaned_text = text_cleaner.clean(text)
        end_time_cleaned_text = time.time()
        elapsed_time_cleaned_text = end_time_cleaned_text - start_time_text_processing

        start_time_corrige = time.time()
        corrected_text = cleaned_text
        # corrected_text, _ = fix_text_with_corrige(cleaned_text, ner=True)
        end_time_corrige = time.time()
        elapsed_time_corrige = end_time_corrige - start_time_corrige

        logger.info("Content Proposition")
        start_time_content_proposition = time.time()
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "url": text
        }
        start_time_ner = time.time()
        ners = ray.get(shared_models.extract_ner.remote(corrected_text, threshold=local_config["NLPAlgorithm_ner_th"]))    
        logger.info(f"NER enrinched content proposition = {ners}")
        end_time_ner = time.time()
        elapsed_time_ner = end_time_ner - start_time_ner

        logger.info("Encyclopedic Content")
        start_time_encyclopedic_content = time.time()

        encyclopedic_content = None
        topic_ner = None
        if global_config["encyclopedic_content"]:
            encyclopedic_content, topic_ner = data_extractor.extract_encyclopedic_content_proposition("",
                                                                                                    corrected_text,
                                                                                                    text_and_metadata["all_metadata"],
                                                                                                    is_webmaster=False)
        end_time_encyclopedic_content = time.time()
        elapsed_time_encyclopedic = end_time_encyclopedic_content - start_time_encyclopedic_content
        logger.info(f"encyclopedic_ontent enrinched content proposition = {encyclopedic_content}")
        progressive_elapsed_time_encyclopedic = end_time_encyclopedic_content - start_time
        progressive_timings["encyclopedic_content"] = progressive_elapsed_time_encyclopedic
        logger.info(f"progressive encyclopedic_content_time = {end_time_encyclopedic_content - start_time}")
        # Preparing the data_encyclopedic_content dictionary
        data_encyclopedic_content = {
            "encyclopedic_content": {
                "content": empty_encyclopedic_content if encyclopedic_content is None else encyclopedic_content,
                "topic": "" if topic_ner is None else topic_ner,
                "liked": False
            }
        }
        data_encyclopedic_content.update(data)
        mqtt_publisher.publish_content(local_config["publisher_ip"],
                                       local_config["publisher_port_enriched_content"],
                                       f"/nlp-response/{user_id}/encyclopedic_content",
                                       data_encyclopedic_content)
        logger.info("Encyclopedic content successfully published")
        start_time_extract_keywords = time.time()
        keywords = ray.get(nlp_algorithm.extract_keywords.remote(corrected_text, html_metadata_keywords))# output form:  [(keyword: str, confidence: int, rewarded: Bool), (...)]
        logger.info(f"keywords enrinched content proposition = {keywords}")
        if keywords is None:
            logger.warning(f"Publish empty cause keywords have not been extracted")
            return mqtt_publisher.publish_empty(user_id, session_id, request_id, text, empty_encyclopedic_content)
        end_time_extract_keywords = time.time()
        elapsed_time_extract_keywords = end_time_extract_keywords - start_time_extract_keywords
        progressive_elapsed_time_keywords = end_time_extract_keywords - start_time
        progressive_timings["keywords"] = progressive_elapsed_time_keywords

        start_time_extract_topic = time.time()
        if local_config["keyword_instead_topic"] == False:
            logger.info("Topic Extraction")
            topic = ray.get(nlp_algorithm.extract_topic.remote(corrected_text, keywords)) # output form: (topic: str, confidence: int)        
            if topic is None:
                topic = (keywords[0][0], keywords[0][1])   
        else:
            logger.info("Using first keyword instead of topic")
            topic = (keywords[0][0], keywords[0][1]) # we need this to remove the third elem that keywords have
        logger.info(f"topic enrinched content proposition = {topic}")

        end_time_extract_topic = time.time()
        elapsed_time_extract_topic = end_time_extract_topic - start_time_extract_topic
        
        # user profiling
        if global_config["reinforcement_mechanism"]:
            ray.get(shared_models.update_embedding_global.remote(user_id, topic[0]))
            ray.get(shared_models.update_embedding_session.remote(user_id, session_id, topic[0]))
            ray.get(shared_models.update_topic_count.remote(user_id, topic[0]))

        start_time_faq = time.time()
        faq = {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": "", "liked": False}}
        if global_config["FAQ_content"]:
            faq = ray.get(nlp_algorithm.faq_generation.remote(corrected_text, max_tokens=300, how_many=global_config["FAQ_number"]))
        logger.info(f"FAQ enrinched content proposition = {faq}")
        end_time_faq = time.time()
        elapsed_time_faq = end_time_faq - start_time_faq
        end_time_text_processing = time.time()
        elapsed_time_text_processing = end_time_text_processing - start_time_text_processing
        progressive_elapsed_time_faq = end_time_faq - start_time
        progressive_timings["faq"] = progressive_elapsed_time_faq

        # publishing FAQs
        data_faq = faq.copy()
        data_faq["FAQ"]["topic"] = topic[0]
        data_faq["FAQ"]["liked"] = False
        data_faq.update(data)
        
        
        mqtt_publisher.publish_content(local_config["publisher_ip"], local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/faq", data_faq)
        logger.info("Faq successfully published")

        start_time_predicted_content = time.time()
        previous_topics_visited = ray.get(nlp_algorithm.get_previous_topics_visited.remote(user_id))
        predicted_content = None
        if global_config["reinforcement_mechanism"] and previous_topics_visited:
            predicted_content = ray.get(shared_models.get_predicted_content.remote(previous_topics_visited[0], previous_topics_visited[1], topic[0], how_many=1))
        end_time_predicted_content = time.time()
        elapsed_time_predicted_content = end_time_predicted_content - start_time_predicted_content    
        logger.info(f"predicted_content enrinched content proposition = {predicted_content}\n")

        start_time_suggested_websites = time.time()
        suggested_websites = None
        if global_config["websites_content"]: 
            suggested_websites = ray.get(nlp_algorithm.websites_content_proposition.remote("", corrected_text, predicted_content, how_many=global_config["link_number"]))
      
        logger.info(f"suggested_websites enrinched content proposition = {suggested_websites}\n")
        end_time_suggested_websites = time.time()
        elapsed_time_suggested_websites = end_time_suggested_websites - start_time_suggested_websites
        progressive_elapsed_time_suggested_websites = end_time_suggested_websites - start_time
        progressive_timings["suggested_websites"] = progressive_elapsed_time_suggested_websites
        if suggested_websites: # it can be None
            # publishing website_content
            data_website_content = {
                "website_content": {
                    "content": data_extractor.get_domain(suggested_websites)
                }
            }
            data_website_content.update(data)
            mqtt_publisher.publish_content(local_config["publisher_ip"], local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/website_content", data_website_content)
            logger.info("Website content successfully published")
        else:
            logger.warning("Website content with text failed to publish as it returned None")
        
        logger.info("Text Generation")
        start_time_text_generation = time.time()
        generated_text_based_on_topic = None
        if global_config["generated_content"]:
            generated_text_based_on_topic = ray.get(nlp_algorithm.generate_proposed_content_based_on_topic.remote(corrected_text, max_tokens=local_config["NLPAlgorithm"]["max_tokens"]))
        logger.info(f"generated_text_based_on_topic enrinched content proposition = {generated_text_based_on_topic}")
        end_time_text_generation = time.time()
        elapsed_time_text_generation = end_time_text_generation - start_time_text_generation   
        progressive_elapsed_time_text_generation = end_time_text_generation - start_time
        progressive_timings["text_generation"] = progressive_elapsed_time_text_generation

        #publishing generative_content
        data_generative_content = {
            "generative_content": {"content": "" if generated_text_based_on_topic is None else generated_text_based_on_topic,
                                   "topic": topic[0],
                                   "liked": False}
        }
        data_generative_content.update(data)
        mqtt_publisher.publish_content(local_config["publisher_ip"],
                                       local_config["publisher_port_enriched_content"],
                                       f"/nlp-response/{user_id}/generative_content",
                                       data_generative_content)
        logger.info("Generative content successfully published")

        elaborated_at = datetime.now()

        # publishing /navigation-request-completion
        mqtt_publisher.publish_navigation_request_completion(user_id,
                                                             session_id,
                                                             text,
                                                             local_config["publisher_ip"],
                                                             local_config["publisher_port_kpi"],
                                                             "/navigation-request-completion",
                                                             requested_at,
                                                             elaborated_at)
    
        logger.info("Are text similar")
        start_time_are_text_similar = time.time()
        similar_text_and_encyclopedic_content, similar_text_and_text_gpt = False, False
        if encyclopedic_content is not None:
            similar_text_and_encyclopedic_content = text_cleaner.are_text_similar(corrected_text,
                                                                                  encyclopedic_content["wiki_text"],
                                                                                  local_config["are_text_similar_th"])
        if generated_text_based_on_topic is not None:
            similar_text_and_text_gpt = text_cleaner.are_text_similar(corrected_text,
                                                                      generated_text_based_on_topic,
                                                                      local_config["are_text_similar_th"])
        end_time_are_text_similar = time.time()
        elapsed_time_are_text_similar = end_time_are_text_similar - start_time_are_text_similar

        end_time_content_proposition = time.time()
        elapsed_time_content_proposition = end_time_content_proposition - start_time_content_proposition
        end_time = time.time()
        elapsed_time = end_time - start_time
        progressive_timings["total_elapsed_time"] = elapsed_time

        text_res = {
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "timing": {
                "time_full_pipeline": elapsed_time,
                "time_data_extraction": elapsed_time_data_extraction,
                "time_text_processing": elapsed_time_text_processing,
                "time_cleaned_text": elapsed_time_cleaned_text,
                "time_corrige": elapsed_time_corrige,
                "time_content_proposition": elapsed_time_content_proposition,
                "time_extract_keywords": elapsed_time_extract_keywords,
                "time_extract_topic": elapsed_time_extract_topic,
                "time_ner": elapsed_time_ner,
                "time_faq": elapsed_time_faq,
                "time_predicted_content": elapsed_time_predicted_content,
                "time_text_generation": elapsed_time_text_generation,
                "time_encyclopedic": elapsed_time_encyclopedic,
                "time_suggested_websites": elapsed_time_suggested_websites,
                "time_are_text_similar": elapsed_time_are_text_similar,

            },
            "content_proposition": {
                "keywords": [kw for kw in keywords], # keywords always exist at this point. If not, we publish empty json immediately and the pipeline is stopped
                "topic": topic, # it's always a tuple, worst case scenario it's the first keyword (keywords[0][0], keywords[0][1])
                "ner": [{"entity_group": "", "score": 0, "word": ""}] if ners is None else ners,
                "faq": faq, # it's always a dict in this form {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": ""}}
                "predicted_content": predicted_content, # is worst case is: [(0.99, topic)]
                "generated_text_based_on_topic": "" if generated_text_based_on_topic is None else generated_text_based_on_topic,
                "encyclopedic_content": empty_encyclopedic_content if encyclopedic_content is None else encyclopedic_content,
                "suggested_websites": suggested_websites,
            },
            "similar_text_and_encyclopedic_content": similar_text_and_encyclopedic_content,
            "similar_text_and_text_gpt": similar_text_and_text_gpt
        }

        # save linguistic analysis data
        linguistic_analysis_text_data = {
            "text": text_res["text"],
            "user_id": text_res["user_id"],
            "session_id": text_res["session_id"],
            "request_id": text_res["request_id"],
            "elapsed_time": text_res["timing"]["time_full_pipeline"],
            "keywords": text_res["content_proposition"]["keywords"],
            "faq": text_res["content_proposition"]["faq"],
            "topic": text_res["content_proposition"]["topic"],
            "ner": text_res["content_proposition"]["ner"],
            "timestamp": datetime.now(),
            "progressive_timings": progressive_timings
        }
        
        store_linguistic_analysis_text(LinguisticAnalysisTextModel(**linguistic_analysis_text_data))

        proposed_content = {
            "user_id": text_res["user_id"],
            "session_id": text_res["session_id"],
            "request_id": text_res["request_id"],
            "elapsed_time": text_res["timing"]["time_full_pipeline"],
            "encyclopedic_content": text_res["content_proposition"]["encyclopedic_content"],
            "generative_content": text_res["content_proposition"]["generated_text_based_on_topic"],
            "website_content": text_res["content_proposition"]["suggested_websites"],
            "faq": text_res["content_proposition"]["faq"],
            "timestamp": datetime.now()
        }

        store_enriched_content_text(EnrichedContentModel(**proposed_content))

        # publishing of execution_time
        mqtt_publisher.publish_execution_timings_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/execution-time", text_res)
        
        # publishing of keywords
        mqtt_publisher.publish_keywords_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/algorithm-results/keywords", text_res)

        # publishing topic
        mqtt_publisher.publish_topic_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/algorithm-results/topics", text_res)
        
        if ners:
        # publishing of ners
            mqtt_publisher.publish_ners_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/algorithm-results/ners", text_res)

        return True
    except Exception as e:
        logger.error(f"{e}")
        return False

@ray.remote
def enriched_content_proposal(url:str, user_id:int, session_id:int, request_id:int):
    try:
        logging.basicConfig(filename="./logs/logs_enriched_content_proposal.txt",
                            format='pid(%(process)d) %(asctime)s %(levelname)-8s %(funcName)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info(f"input_url={url}")

        kw_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"
        topic_classifier = f"app/models/{get_global_config()['models']['topic_extraction_algorithm']['model_name']}"

        nlp_algorithm = UnsharedModelsActor.remote(get_local_config(), get_global_config(), kw_model, topic_classifier)

        requested_at = datetime.now()
        progressive_timings = {}
        start_time = time.time()
        global_config = get_global_config()
        local_config = get_local_config()
        text_cleaner = TextCleaner(global_config)
        data_extractor = DataExtractor(local_config)
        mqtt_publisher = MqttPublisher(local_config)
                        
        start_time_data_extraction = time.time()
        url = text_cleaner.process_url(url)
        text_and_metadata = data_extractor.extract_text_and_metadata(url)
        empty_encyclopedic_content = {'wiki_text': '', 'wiki_url': '', 'title_page': '', 'page_description': '', 'page_keywords': []}
        if text_and_metadata is None:
            logger.warning("Cannot retrieve text_and_metadata")
            return mqtt_publisher.publish_empty(user_id, session_id, request_id, url, empty_encyclopedic_content)
        else:
            html_metadata_keywords = text_and_metadata["all_metadata"]["page_keywords"]
            text = text_and_metadata["text"]
        end_time_data_extraction = time.time()
        elapsed_time_data_extraction = end_time_data_extraction - start_time_data_extraction

        logger.info(f"Extracted text: {text}")
        start_time_text_processing = time.time()
        cleaned_text = text_cleaner.clean(text)
        end_time_cleaned_text = time.time()
        elapsed_time_cleaned_text = end_time_cleaned_text - start_time_text_processing

        start_time_corrige = time.time()
        corrected_text = cleaned_text
        # corrected_text, _ = fix_text_with_corrige(cleaned_text, ner=True)
        end_time_corrige = time.time()
        elapsed_time_corrige = end_time_corrige - start_time_corrige



        logger.info("Content Proposition")
        start_time_content_proposition = time.time()
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "url": url
        }

        start_time_ner = time.time()
        ners = ray.get(shared_models.extract_ner.remote(corrected_text, threshold=local_config["NLPAlgorithm_ner_th"])) # "ner": [{"entity_group": "", "score": 0, "word": ""}, ...],
        logger.info(f"NER enrinched content proposition = {ners}")
        end_time_ner = time.time()
        elapsed_time_ner = end_time_ner - start_time_ner

        logger.info("Encyclopedic Content")
        start_time_encyclopedic_content = time.time()

        encyclopedic_content = None
        topic_ner = None
        
        if global_config["encyclopedic_content"]:
            encyclopedic_content, topic_ner = data_extractor.extract_encyclopedic_content_proposition(url,
                                                                                                    corrected_text,
                                                                                                    text_and_metadata["all_metadata"],
                                                                                                    is_webmaster=False)
        print(f"encyclopedic_content, topic_ner: {encyclopedic_content} and {topic_ner}")
        end_time_encyclopedic_content = time.time()
        elapsed_time_encyclopedic = end_time_encyclopedic_content - start_time_encyclopedic_content
        logger.info(f"encyclopedic_content enrinched content proposition = {encyclopedic_content}")
        progressive_elapsed_time_encyclopedic = end_time_encyclopedic_content - start_time
        progressive_timings["encyclopedic_content"] = progressive_elapsed_time_encyclopedic
        logger.info(f"progressive encyclopedic_content_time = {end_time_encyclopedic_content - start_time}")
        # Preparing the data_encyclopedic_content dictionary
        data_encyclopedic_content = {
            "encyclopedic_content": {
                "content": empty_encyclopedic_content if encyclopedic_content is None else encyclopedic_content,
                "topic": "" if topic_ner is None else topic_ner,
                "liked": False
            }
        }
        data_encyclopedic_content.update(data)
        mqtt_publisher.publish_content(local_config["publisher_ip"], local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/encyclopedic_content", data_encyclopedic_content)
        logger.info("Encyclopedic content successfully published")

        start_time_extract_keywords = time.time()
        keywords = ray.get(nlp_algorithm.extract_keywords.remote(corrected_text, html_metadata_keywords)) # output form:  [(keyword: str, confidence: int, rewarded: Bool), (...)]
        logger.info(f"keywords enrinched content proposition = {keywords}\n")
        if keywords is None:
            logger.warning(f"Publish empty cause keywords have not been extracted")
            return mqtt_publisher.publish_empty(user_id, session_id, request_id, url, empty_encyclopedic_content)
        end_time_extract_keywords = time.time()
        elapsed_time_extract_keywords = end_time_extract_keywords - start_time_extract_keywords
        progressive_elapsed_time_keywords = end_time_extract_keywords - start_time
        progressive_timings["keywords"] = progressive_elapsed_time_keywords

        start_time_extract_topic = time.time()
        if local_config["keyword_instead_topic"] == False:
            logger.info("Topic Extraction")
            topic = ray.get(nlp_algorithm.extract_topic.remote(corrected_text, keywords)) # output form: (topic: str, confidence: int)        
            if topic is None:
                topic = (keywords[0][0], keywords[0][1])   
        else:
            logger.info("Using first keyword instead of topic")
            topic = (keywords[0][0], keywords[0][1]) # we need this to remove the third elem that keywords have
        logger.info(f"Topic enrinched content proposition = {topic}")

        end_time_extract_topic = time.time()
        elapsed_time_extract_topic = end_time_extract_topic - start_time_extract_topic
        
        # user profiling
        if global_config["reinforcement_mechanism"]:
            ray.get(shared_models.update_embedding_global.remote(user_id, topic[0]))
            ray.get(shared_models.update_embedding_session.remote(user_id, session_id, topic[0]))
            ray.get(shared_models.update_topic_count.remote(user_id, topic[0]))

        start_time_faq = time.time()
        faq = {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": ""}}
        if global_config["FAQ_content"]:
            faq = ray.get(nlp_algorithm.faq_generation.remote(corrected_text, max_tokens=300, how_many=global_config["FAQ_number"]))
        logger.info(f"FAQ enrinched content proposition = {faq}\n")
        end_time_faq = time.time()
        elapsed_time_faq = end_time_faq - start_time_faq
        end_time_text_processing = time.time()
        elapsed_time_text_processing = end_time_text_processing - start_time_text_processing
        progressive_elapsed_time_faq = end_time_faq - start_time
        progressive_timings["faq"] = progressive_elapsed_time_faq

        # publishing FAQs
        data_faq = faq.copy()
        data_faq["FAQ"]["topic"] = topic[0]
        data_faq["FAQ"]["liked"] = False
        data_faq.update(data)
        
        mqtt_publisher.publish_content(local_config["publisher_ip"], local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/faq", data_faq)
        logger.info("Faq successfully published")


        start_time_predicted_content = time.time()
        previous_topics_visited = ray.get(nlp_algorithm.get_previous_topics_visited.remote(user_id))
        predicted_content = None
        if global_config["reinforcement_mechanism"] and previous_topics_visited:
            predicted_content = ray.get(shared_models.get_predicted_content.remote(previous_topics_visited[0], previous_topics_visited[1], topic[0], how_many=1))
        end_time_predicted_content = time.time()
        elapsed_time_predicted_content = end_time_predicted_content - start_time_predicted_content    
        logger.info(f"predicted_content enrinched content proposition = {predicted_content}")
        
        start_time_suggested_websites = time.time()
        suggested_websites = None
        if global_config["websites_content"]:
            suggested_websites = ray.get(nlp_algorithm.websites_content_proposition.remote(url, corrected_text, predicted_content, how_many=global_config["link_number"]))
        logger.info(f"suggested_websites enrinched content proposition = {suggested_websites}")
        end_time_suggested_websites = time.time()
        elapsed_time_suggested_websites = end_time_suggested_websites - start_time_suggested_websites
        progressive_elapsed_time_suggested_websites = end_time_suggested_websites - start_time
        progressive_timings["suggested_websites"] = progressive_elapsed_time_suggested_websites

        if suggested_websites: # it can be None
            # publishing website_content
            data_website_content = {
                "website_content": {
                    "content": data_extractor.get_domain(suggested_websites)
                }
            }
            data_website_content.update(data)
            mqtt_publisher.publish_content(local_config["publisher_ip"], local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/website_content", data_website_content)
            logger.info("Website content successfully published")
        else:
            logger.warning("Website content with text failed to publish as it returned None")
        
        logger.info("Text Generation")
        start_time_text_generation = time.time()
        generated_text_based_on_topic = None
        if global_config["generated_content"]:
            generated_text_based_on_topic = ray.get(nlp_algorithm.generate_proposed_content_based_on_topic.remote(corrected_text, max_tokens=local_config["NLPAlgorithm"]["max_tokens"]))
        logger.info(f"generated_text_based_on_topic enrinched content proposition = {generated_text_based_on_topic}")
        end_time_text_generation = time.time()
        elapsed_time_text_generation = end_time_text_generation - start_time_text_generation   
        progressive_elapsed_time_text_generation = end_time_text_generation - start_time
        progressive_timings["text_generation"] = progressive_elapsed_time_text_generation

        #publishing generative_content
        data_generative_content = {
            "generative_content": {"content": "" if generated_text_based_on_topic is None else generated_text_based_on_topic,
                                   "topic": topic[0],
                                   "liked": False}
        }
        data_generative_content.update(data)
        mqtt_publisher.publish_content(local_config["publisher_ip"], local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/generative_content", data_generative_content)
        logger.info("Generative content successfully published")

        elaborated_at = datetime.now()

        # publishing /navigation-request-completion
        mqtt_publisher.publish_navigation_request_completion(user_id, session_id, url, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/navigation-request-completion", requested_at, elaborated_at)
    
        logger.info("Are text similar")
        start_time_are_text_similar = time.time()
        similar_text_and_encyclopedic_content, similar_text_and_text_gpt = False, False
        if encyclopedic_content is not None:
            similar_text_and_encyclopedic_content = text_cleaner.are_text_similar(corrected_text,
                                                                                  encyclopedic_content["wiki_text"],
                                                                                  local_config["are_text_similar_th"])
        if generated_text_based_on_topic is not None:
            similar_text_and_text_gpt = text_cleaner.are_text_similar(corrected_text,
                                                                      generated_text_based_on_topic,
                                                                      local_config["are_text_similar_th"])
        end_time_are_text_similar = time.time()
        elapsed_time_are_text_similar = end_time_are_text_similar - start_time_are_text_similar

        end_time_content_proposition = time.time()
        elapsed_time_content_proposition = end_time_content_proposition - start_time_content_proposition
        end_time = time.time()
        elapsed_time = end_time - start_time
        progressive_timings["total_elapsed_time"] = elapsed_time

        res = {
            "url": url,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "timing": {
                "time_full_pipeline": elapsed_time,
                "time_data_extraction": elapsed_time_data_extraction,
                "time_text_processing": elapsed_time_text_processing,
                "time_cleaned_text": elapsed_time_cleaned_text,
                "time_corrige": elapsed_time_corrige,
                "time_content_proposition": elapsed_time_content_proposition,
                "time_extract_keywords": elapsed_time_extract_keywords,
                "time_extract_topic": elapsed_time_extract_topic,
                "time_ner": elapsed_time_ner,
                "time_faq": elapsed_time_faq,
                "time_predicted_content": elapsed_time_predicted_content,
                "time_text_generation": elapsed_time_text_generation,
                "time_encyclopedic": elapsed_time_encyclopedic,
                "time_suggested_websites": elapsed_time_suggested_websites,
                "time_are_text_similar": elapsed_time_are_text_similar,

            },
            "content_proposition": {
                "keywords": [kw for kw in keywords], # keywords always exist at this point. If not, we publish empty json immediately and the pipeline is stopped
                "topic": topic, # it's always a tuple, worst case scenario it's the first keyword (keywords[0][0], keywords[0][1])
                "ner": [{"entity_group": "", "score": 0, "word": ""}] if ners is None else ners,
                "faq": faq, # it's always a dict in this form {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": ""}}
                "predicted_content": predicted_content, # is worst case is: [(0.99, topic)]
                "generated_text_based_on_topic": "" if generated_text_based_on_topic is None else generated_text_based_on_topic,
                "encyclopedic_content": empty_encyclopedic_content if encyclopedic_content is None else encyclopedic_content,
                "suggested_websites": suggested_websites,
            },
            "similar_text_and_encyclopedic_content": similar_text_and_encyclopedic_content,
            "similar_text_and_text_gpt": similar_text_and_text_gpt
        }

        # save linguistic analysis data
        linguistic_analysis_data = {
            "url": url,
            "user_id": res["user_id"],
            "session_id": res["session_id"],
            "request_id": res["request_id"],
            "elapsed_time": res["timing"]["time_full_pipeline"],
            "keywords": res["content_proposition"]["keywords"],
            "faq": res["content_proposition"]["faq"],
            "topic": res["content_proposition"]["topic"],
            "ner": res["content_proposition"]["ner"],
            "timestamp": datetime.now(),
            "progressive_timings": progressive_timings
        }
        
        store_linguistic_analysis(LinguisticAnalysisModel(**linguistic_analysis_data))

        proposed_content = {
            "user_id": res["user_id"],
            "session_id": res["session_id"],
            "request_id": res["request_id"],
            "elapsed_time": res["timing"]["time_full_pipeline"],
            "encyclopedic_content": res["content_proposition"]["encyclopedic_content"],
            "generative_content": res["content_proposition"]["generated_text_based_on_topic"],
            "website_content": res["content_proposition"]["suggested_websites"],
            "faq": res["content_proposition"]["faq"],
            "timestamp": datetime.now()
        }

        store_enriched_content(EnrichedContentModel(**proposed_content))

        # publishing of execution_time
        mqtt_publisher.publish_execution_timings_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/execution-time", res)
        
        # publishing of keywords
        mqtt_publisher.publish_keywords_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/algorithm-results/keywords", res)

        # publishing topic
        mqtt_publisher.publish_topic_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/algorithm-results/topics", res)
        
        if ners:
        # publishing of ners
            mqtt_publisher.publish_ners_kpi(user_id, local_config["publisher_ip"], local_config["publisher_port_kpi"], "/algorithm-results/ners", res)

        return True
    except Exception as e:
        logger.error(f"{e}")
        return False
    

@ray.remote
def webmaster_content_proposal(webpage_url: str):
    try:
        logging.basicConfig(filename="./logs/logs_webmaster_content_proposal.txt",
                            format='pid(%(process)d) %(asctime)s %(levelname)-8s %(funcName)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info(f"url = {webpage_url}")
        global_config = get_global_config()
        local_config = get_local_config()
        text_cleaner = TextCleaner(global_config)
        data_extractor = DataExtractor(local_config)
        mqtt_publisher = MqttPublisher(local_config)
        formatter = Formatter()

        kw_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"
        topic_classifier = f"app/models/{get_global_config()['models']['topic_extraction_algorithm']['model_name']}"

        nlp_algorithm = UnsharedModelsActor.remote(get_local_config(), get_global_config(), kw_model, topic_classifier)
        webpage_url = text_cleaner.process_url(webpage_url)
        text_and_metadata = data_extractor.extract_text_and_metadata(webpage_url)
        if text_and_metadata is None:
            logger.warning("Publish empty webmaster")
            return mqtt_publisher.publish_empty_webmaster(webpage_url, "46.137.91.119", 1883, "/webmaster_extraction")
        else:
            all_metadata = text_and_metadata["all_metadata"]
            text = text_and_metadata["text"]
        try:
            words_count_in_text = len(text.split())
        except Exception as e:
            logger.warning(f"words_count_in_text: {e}")
            words_count_in_text = 0

        cleaned_text = text_cleaner.clean(text)
        corrected_text, corrige_corrections = fix_text_with_corrige(cleaned_text, ner=True)
        ners = ray.get(shared_models.extract_ner.remote(corrected_text, threshold=local_config["NLPAlgorithm_ner_th"]))
        encyclopedic_content = None
        if global_config["encyclopedic_content"]:
            encyclopedic_content, _ = data_extractor.extract_encyclopedic_content_proposition(webpage_url,
                                                                                                  corrected_text,
                                                                                                  all_metadata,
                                                                                                  is_webmaster=True)
        
        logger.info(f"Encyclopedic content webmaster: {encyclopedic_content}")
        keywords = ray.get(nlp_algorithm.extract_keywords.remote(corrected_text, all_metadata["page_keywords"])) 
        if keywords is None:
            logger.warning(f"Publish empty webmaster cause keywords have not been extracted")
            return mqtt_publisher.publish_empty_webmaster(webpage_url, "46.137.91.119", 1883, "/webmaster_extraction")
        logger.info(f"Keywords webmaster: {keywords}")
        
        if local_config["keyword_instead_topic"] == False:
            topic = ray.get(nlp_algorithm.extract_topic.remote(corrected_text, keywords))
            if topic is None:
                topic = (keywords[0][0], keywords[0][1])   
        else:
            logger.info("Using first keyword instead of topic")
            topic = (keywords[0][0], keywords[0][1]) # we need this to remove the third elem that keywords have. Note that keyword will never be None cause if is would be, we return above the publish empty
        logger.info(f"Topic webmaster: {topic}")

        faq = {}
        if global_config["FAQ_content"]:
            faq = ray.get(nlp_algorithm.faq_generation.remote(corrected_text, max_tokens=local_config["NLPAlgorithm"]["max_tokens"], how_many=global_config["FAQ_number"]))
        logger.info(f"FAQ webmaster: {faq}")
        
        suggested_websites = None
        if global_config["websites_content"]:
            suggested_websites = ray.get(nlp_algorithm.websites_content_proposition.remote(webpage_url, corrected_text, None, how_many=global_config["link_number"]))
        logger.info(f"Suggested Website webmaster: {suggested_websites}")
        
        generated_text_based_on_topic = None
        if global_config["generated_content"]:
            generated_text_based_on_topic = ray.get(nlp_algorithm.generate_proposed_content_based_on_topic.remote(corrected_text, max_tokens=local_config["NLPAlgorithm"]["max_tokens"]))
        logger.info(f"Generative Content webmaster: {generated_text_based_on_topic}")
        
        webmaster_content = {
            "webpage_url" : webpage_url,
            "encyclopedic_content": "" if encyclopedic_content is None else encyclopedic_content['wiki_text'],
            "generative_content": "" if generated_text_based_on_topic is None else generated_text_based_on_topic,
            "website_content": formatter.refactor_website_content_format(suggested_websites),
            "faqs": formatter.refactor_faq_format(faq['FAQ']['content']),
            "keywords": formatter.refactor_keywords_format(keywords),
            "topics": formatter.refactor_topic_format(topic),
            "ners":  [] if ners is None else ners,
            "corrige_corrections": [] if corrige_corrections is None else corrige_corrections,
            "webpage_text": "" if text is None else text,
            "word_count": words_count_in_text,
            "title_page": ""  if all_metadata["title_page"] is None else all_metadata["title_page"],
            "page_description": ""  if all_metadata["page_description"] is None else all_metadata["page_description"],
            "page_keywords": []  if all_metadata["page_keywords"] is None else all_metadata["page_keywords"],
            "page_author": ""  if all_metadata["page_author"] is None else all_metadata["page_author"],
            "page_lang": ""  if all_metadata["page_lang"] is None else all_metadata["page_lang"]
        }
        
        mqtt_publisher.publish_content("46.137.91.119", 1883, f"/webmaster_extraction", webmaster_content)
        
        store_webmaster_content(WebmasterContentModel(**webmaster_content))
        return True
    except Exception as e:
        logger.error(f"{e}")
        return False


@ray.remote
def delete_model_task(user_model_name: str):
    kw_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"
    topic_classifier = f"app/models/{get_global_config()['models']['topic_extraction_algorithm']['model_name']}"
    nlp_algorithm = UnsharedModelsActor.remote(get_local_config(), get_global_config(), kw_model, topic_classifier)
    if ray.get(nlp_algorithm.delete_model.remote(user_model_name)) is None:
        return False
    return True




# TODO: finish this once we have url
@ray.remote
def update_embedding_feedback_task(user_feedback):
    try:
        ray.get(shared_models.update_embedding_global.remote(user_feedback["user_id"], user_feedback["content_topic"], is_feedback=True))
        ray.get(shared_models.update_embedding_session.remote(user_feedback["user_id"], user_feedback["session_id"], user_feedback["content_topic"], is_feedback=True))
        store_user_feedback(UserFeedbackTopicModel(**user_feedback))
        #hardcoded_url = "https://it.wikipedia.org/wiki/Ravenna"
        mqtt_publisher = MqttPublisher(get_local_config())

        data = {
            "user_id": user_feedback["user_id"],
            "session_id": user_feedback["session_id"],
            "request_id": user_feedback["request_id"],
            "url": user_feedback["url"]
        }

        if user_feedback["content_type"] == "encyclopedic_content":
            user_feedback["data"]["liked"] = True
            new_dict = {"encyclopedic_content": user_feedback["data"]}
            new_dict.update(data)
            mqtt_publisher.publish_content(get_local_config()["publisher_ip"], get_local_config()["publisher_port_enriched_content"], f"/nlp-response/{user_feedback['user_id']}/encyclopedic_content", new_dict)
        elif user_feedback["content_type"] == "FAQ":
            user_feedback["data"]["liked"] = True
            new_dict = {"FAQ": user_feedback["data"]}
            new_dict.update(data)
            mqtt_publisher.publish_content(get_local_config()["publisher_ip"], get_local_config()["publisher_port_enriched_content"], f"/nlp-response/{user_feedback['user_id']}/faq", new_dict)
        elif user_feedback["content_type"] == "website_content":
            user_feedback["data"]["liked"] = True
            new_dict = {"website_content": user_feedback["data"]}
            new_dict.update(data)
            mqtt_publisher.publish_content(get_local_config()["publisher_ip"], get_local_config()["publisher_port_enriched_content"], f"/nlp-response/{user_feedback['user_id']}/website_content", new_dict)
        elif user_feedback["content_type"] == "generative_content":
            user_feedback["data"]["liked"] = True
            new_dict = {"generative_content": user_feedback["data"]}
            new_dict.update(data)
            mqtt_publisher.publish_content(get_local_config()["publisher_ip"], get_local_config()["publisher_port_enriched_content"], f"/nlp-response/{user_feedback['user_id']}/generative_content", new_dict)
        
        return True
    except Exception as e:
        print(f"Error in user_feedback: {e}\n", flush=True)
        return False

@ray.remote
def load_model_task(user_model_name: str):
    try:
        kw_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"
        topic_classifier = f"app/models/{get_global_config()['models']['topic_extraction_algorithm']['model_name']}"
        nlp_algorithm = UnsharedModelsActor.remote(get_local_config(), get_global_config(), kw_model, topic_classifier)
        if ray.get(nlp_algorithm.download_and_save_model_from_url.remote(user_model_name)) is None:
            return False
        return True
    except Exception as e:
        print(f"Error in load_model: {e}\n", flush=True)
        return False

# NOTE: mongo does not like @ray.remote
def user_delete_task(user_id: int):
    try:
        # ASSUMPTION: if user_id is present in `user_profiling_global_collection`, we are sure that it is present also in `user_profiling_session_collection` and `user_profiling_topic_count_global_collection`. 
        global_deletion_result = user_profiling_global_collection.delete_one({"user_id": user_id})        
        session_deletion_result = user_profiling_session_collection.delete_many({"user_id": user_id})
        topic_count_global_deletion_result = user_profiling_topic_count_global_collection.delete_one({"user_id": user_id})
        if global_deletion_result.deleted_count != 1:
            return False
        return True
    except Exception as e:
        print(f"Error in user_delete: {e}\n", flush=True)
        return False
    


