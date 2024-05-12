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
from sentence_transformers import SentenceTransformer, util
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
import numpy as np
from database.db import create_user_profiling_global, update_user_profiling_global, get_user_global
from database.db import create_user_profiling_session, update_user_profiling_session, get_user_session
from database.db import create_user_profiling_topic_count_global, update_user_profiling_topic_count_global, get_user_topic_count_global
from app.mongo_models.user_profiling_global_model import UserProfilingGlobalModel
from app.mongo_models.user_profiling_session_model import UserProfilingSessionModel
from app.mongo_models.user_profiling_topic_count_global_model import UserProfilingTopicCountGlobalModel
from datetime import datetime
import os
import shutil
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
import time
from urllib.parse import unquote
from gensim.models import FastText

device_type = 0 if torch.cuda.is_available() else -1

# Initialize Ray
ray.init(num_cpus=None)

@ray.remote
class UnsharedModelsActor:
    def __init__(self, local_config, global_config, kw_model, topic_classifier) -> None:
        self.kw_model = KeyBERT(model=kw_model)
        self.local_config = local_config
        self.global_config = global_config
        if self.local_config["keyword_instead_topic"] is True:
            self.topic_classifier = None
        else:
            self.topic_classifier = pipeline("zero-shot-classification",
                                             model=topic_classifier,
                                             device=device_type, use_fast=True, multi_label=False)
        self.__api_key = self.local_config["openai_api_key"]
        self.google_search_api_key = self.local_config["google_search_api_key"]
        
    def faq_format(self, faq):
        """
        faq example: {"question": "answer"}
        return example: {"FAQ": {"content": ["question": "answer to the first question", "question": "answer to the second question", ...]}}
        """
        list_of_dicts = []
        if not faq:
            data_faq = {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": ""}}
        else:
            for question, answer in faq.items():
                list_of_dicts.append({"question": question, "answer": answer})
            data_faq = {
                "FAQ": {
                    "content": list_of_dicts
                }
            }
        return data_faq

    def faq_generation(self, corrected_text, max_tokens, how_many):
        if how_many < 1:
            return self.faq_format(None)
        if max_tokens < 1:
            return self.faq_format(None)
        # Construct a prompt for GPT to generate questions from keywords and provide answers in JSON format  
        prompt = (
            f"""
            Crea un JSON dove la chiave è la domanda, il valore è la risposta, voglio {how_many} coppie. Concludi ogni risposta con un punto e doppie virgolette. Ogni valore del JSON ha un lunghezza massima di {max_tokens} tokens.
            Un esempio di output è il seguente, ma fai attenzione a non darmi questo come output e restituicimi solamente il json in output.
            Non fornire nessun testo ulteriore, ma solo il JSON.
            Se il testo parla di cookies, pubblicità o abbonamenti, non considerlo ma concentrati sul focus principale.
            L'informazione su cui ti devi basare è la seguente: '{corrected_text}'.
            """
        )

        try:
            client = OpenAI(api_key=self.__api_key)
        
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You generate only the json, nothing more. This is the format {{\"Domanda inerente al contesto\":\"Risposta inerente al contesto\"}}"},
                {"role": "user", "content": prompt}
                ],
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"Error using OpenAI api with error {e}", flush=True)
            return self.faq_format(None)
        try:
            response = ast.literal_eval(response)
        except Exception as e:
            print(f"Error in ast.literal_eval in faq_generation, GPT response: {response}", flush=True)
            return self.faq_format(None)

        # removing double quotes
        response_wihtout_double_quotes = {}
        for key, value in response.items():
            key = key.replace("\"", "'")
            value = value.replace("\"", "'")
            response_wihtout_double_quotes[key] = value
        try:
            limited_faq = dict(list(response_wihtout_double_quotes.items())[:how_many]) # needed to cut the dict because sometimes gpt fails to understand the prompt and generate more results.
            return self.faq_format(limited_faq)
        except json.JSONDecodeError as e:
            return self.faq_format(None)

    def __trim_response(self, chatgpt_response):
        if "Mi dispiace" in chatgpt_response: # Sometimes GPT Gives answers like "Mi dispiace, ...". We have to remove these cases.
            return None
        if "." in chatgpt_response and chatgpt_response[-1] != ".": # this means that the chatgpt_response has not ended and is trimmed
            last_dot_pos = chatgpt_response.rfind(".")
            return chatgpt_response[:last_dot_pos] + "." # end with a dot the sentence
        else:
            return chatgpt_response

    def generate_proposed_content_based_on_topic(self, corrected_text, max_tokens):
        if max_tokens < 1:
            return None
        
        prompt_for_chatgpt = f"""
        Generami un contenuto in italiano basandoti sul testo che ti passerò.
        Non generare nient'altro in output, solo il testo generato.
        Sii prolisso.
        Se il testo parla di cookies, pubblicità o abbonamenti, non considerlo ma concentrati sul focus principale del testo.
        Finisci con un punto e considera che hai al massimo {max_tokens} tokens.
        Ecco il testo: {corrected_text}.
        """
        try:   
            client = OpenAI(api_key=self.__api_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un esperto capace di rispondere a qualsiasi domanda, generi solo la risposta, senza interloquire con l'utente."},
                    {"role": "user", "content": prompt_for_chatgpt}
                ],
                max_tokens=max_tokens
            )
            chatgpt_response = completion.choices[0].message
            trimmed_gpt_response = self.__trim_response(chatgpt_response.content)
            return trimmed_gpt_response
        except Exception as e:
            print(f"Error using OpenAI api with error {e}", flush=True)
            return None
        

    def websites_content_proposition(self, url, extracted_text, predicted_content, how_many) -> dict:
        if predicted_content and predicted_content[0] != 0.99: # in predicted content function we return [(0.99, topic)] bu default if there is an error. Also, `predicted_content` can be None if `reinforcement_mechanism` is deactivated
            how_many = max(0, how_many-1) # we do this to avoid `how_many` to be -1 after we do a decrease by 1
        # if url is wikipedia, use the word as extracted_text
        if "it.wikipedia.org/wiki/" in url:
            extracted_text = url.split("/")[-1] 
        proposed_sites = {}
        params = {
            'key': self.local_config["google_search_api_key"],
            'cx': self.local_config["NLPAlgorithm"]["google_cse_allwebsites_id"],
            'q': extracted_text[:self.local_config["NLPAlgorithm"]["chars_to_consider_allwebsites"]] # TODO: trim at last dot
        }
        try: 
            response = requests.get(self.local_config["NLPAlgorithm"]["base_google_url"],
                                    allow_redirects=True,
                                    params=params,
                                    headers=self.local_config["DataExtractor"]["get_request_header"],
                                    timeout=self.local_config["DataExtractor"]["get_request_timeout"])
        except Exception as e:
            print(f"Exception in request.get(): {e}", flush=True)
            return []
        if response.status_code != 200:
            print(f"Get request failed with status_code {response.status_code}", flush=True)
            return []
        data = response.json()
        items = data.get("items", [])
        if not items:
            return proposed_sites
        
        links = [[item.get("title"), unquote(item.get("link"))] for item in items if unquote(item.get("link")) != url][:how_many]
        for i in range(len(links)):
            proposed_sites[links[i][0]] = links[i][1]

        # get url based on predicted content
        if predicted_content and predicted_content[0] != 0.99:
            params = {
                'key': self.local_config["google_search_api_key"],
                'cx': self.local_config["NLPAlgorithm"]["google_cse_allwebsites_id"],
                'q': predicted_content
            }
            try: 
                response = requests.get(self.local_config["NLPAlgorithm"]["base_google_url"],
                                        allow_redirects=True,
                                        params=params,
                                        headers=self.local_config["DataExtractor"]["get_request_header"],
                                        timeout=self.local_config["DataExtractor"]["get_request_timeout"])
            except Exception as e:
                print(f"Exception in request.get(): {e}", flush=True)
                return []
            if response.status_code != 200:
                print(f"Get request failed with status_code {response.status_code}", flush=True)
                return []
            data = response.json()
            items = data.get("items", [])
            links = [[item.get("title"), unquote(item.get("link"))] for item in items if unquote(item.get("link")) != url][:1]
            for i in range(len(links)):
                proposed_sites[links[i][0]] = links[i][1]
                break
        return proposed_sites


    def get_previous_topics_visited(self, user_id) -> bool:
        user = get_user_topic_count_global(user_id)
        previous_topics_visited = []
        if user:
            if len(user["user_history"]) >= 2: # this is always 2 because `get_predicted_content()` need two args always and the third is alwaus the topic[0]
                sorted_topics = sorted(user["user_history"], key=lambda k: k["last_visit"].isoformat(), reverse=True)[:2]
                return [sorted_topics[0]["topic_name"], sorted_topics[1]["topic_name"]]
        return previous_topics_visited


    def __update_keywords_score(self, keywords, metadata): # output is a list of tuple of triplets
        lower_case_keywords = [kw[0].lower() for kw in keywords]
        lower_case_metadata = set([m.lower() for m in metadata])
        for idx, key in enumerate(lower_case_keywords):
            if key in lower_case_metadata:
                keywords[idx][1] += 0.1 * keywords[idx][1]
                keywords[idx].append(True)
            else:
                keywords[idx].append(False)
        return keywords

    def extract_keywords(self, text: str, metadata):
        try:
            keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), diversity=0.5, use_mmr=True)
            if not keywords:  # check if keywords list is empty or None
                print(f"KeyBERT returned an empty list of keywords or is None.", flush=True)
                return None
            keywords = [list(kw) for kw in keywords]
        except Exception as e:
            print(f"Error in extract keywords, {e}", flush=True)
            return None
        if metadata is None:
            return [kw + [False] for kw in keywords]
        else:
            return self.__update_keywords_score(keywords, metadata)


    def extract_topic(self, text, keywords):
        """
        keywords example: [[key1, 0.7, True], [key2, 0.6, False], [key3, 0.4, True]]
        return example: ("some topic", 0.5)
        """
        candidate_labels = []
        for el in keywords:
            candidate_labels.append(el[0])
        hypothesis_template = "si parla di {}"
        try:
            res = self.topic_classifier(text, candidate_labels, hypothesis_template=hypothesis_template) # classifier can fail if the input text is invalid, very unlikely, but possible
        except Exception as e:
            print(f"Error in Topic extraction: {e}", flush=True)
            return None
        out = []
        labels = res['labels']
        scores = res['scores']
        for idx, val in enumerate(labels):
            out.append((val, scores[idx]))
            break # we break because we want just the first keyword (namely, the topic)
        return out[0] # if we reach this point, out will always have one element
    
    def download_and_save_model_from_url(self, user_model_name):
        try:
            # Use a temporary directory for caching the model
            with tempfile.TemporaryDirectory() as temp_cache_dir:
                model = AutoModel.from_pretrained(user_model_name, cache_dir=temp_cache_dir)
                model.save_pretrained(self.local_config["NLPAlgorithm"]["path_models_download"] + "/" + user_model_name)
                print("Model saved to " + self.local_config["NLPAlgorithm"]["path_models_download"] + "/" + user_model_name, flush=True)
            return user_model_name
        except Exception as e:
            print(f"Failed to download and save model with error: {e}", flush=True)
            return None

    def delete_model(self, user_model_name:str):
        """
        user_model_name: "username/modelname"
        """
        user_name = user_model_name[:user_model_name.find("/")]
        user_folder = self.local_config["NLPAlgorithm"]["path_models_download"] + "/" + user_name
        model_name = user_model_name[user_model_name.find("/")+1:]
        if os.path.isdir(user_folder + "/" + model_name):
            shutil.rmtree(user_folder + "/" + model_name)
        else:
            print("model directory does not exist", flush=True)
            return None
        if not os.listdir(user_folder):
            shutil.rmtree(user_folder)
        return model_name


@ray.remote
class SharedModelsActor:
    def __init__(self, fasttext_model, gensim_fasttext_model, ner_model, ner_tokenizer) -> None:
        self.fasttext_model = fasttext.load_model(fasttext_model)
        self.gensim_fasttext_model = FastText.load_fasttext_format(fasttext_model)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model)
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_tokenizer)

    def __uniform_ners(self, ners):
        for ner in ners:
            ner["word"] = ner["word"].replace("-", "") # the "Coca - Cola" case
        return ners
    
    def extract_ner(self, text, threshold):
        """
        return example: [{"entity_group": "LOC", "score": 0.99, "word": "Santa Maria del Monte"}, ...]
        """
        try:
            nlp = pipeline("ner", model=self.ner_model, device=device_type, tokenizer=self.ner_tokenizer, aggregation_strategy="simple")
            ners = nlp(text)
        except Exception as e:
            print(f"Failed to extract ner with error {e}", flush=True)
            return None

        if not isinstance(ners, list):
            print("ners is not a list", flush=True)
            return None

        if ners == []:
            return None
        
        if ('word' not in ners[0]) or ('score' not in ners[0]):
            return None

        for ner in ners:
            if 'start' in ner:
                del ner['start']
            if 'end' in ner:
                del ner['end']
        # remove NER duplicates
        word_set = set()
        out_ners = []
        for ner in ners:
            ner['score'] = ner['score'].item()
            if ner['word'].startswith('##') or (ner['word'] in word_set) or (ner['score'] <= threshold):
                continue
            word_set.add(ner['word'])
            out_ners.append(ner)
        out_ners.sort(key=lambda x: x['score'], reverse=True)
        out_ners = self.__uniform_ners(out_ners)
        return None if out_ners == [] else out_ners

        
    def get_word_vector(self, word):
        return self.fasttext_model.get_word_vector(word)

    # NOTE: not used as it is not working well at this task
    def get_closest_string_from_embeddings(self, user_embedding):

        def find_closest_string(embedding, topn=1):
            # Find top-N most similar keys by vector
            similar_keys = self.gensim_fasttext_model.wv.most_similar(positive=[embedding], topn=topn)
            
            # Extract strings from the similar keys
            closest_strings = [key for key, _ in similar_keys]
            
            return closest_strings[0] if topn == 1 else closest_strings

        closest_string = find_closest_string(user_embedding)

        return closest_string



    def get_predicted_content(self, param1: str, param2: str, topic: str, how_many:int):
        try:
            results = self.fasttext_model.get_analogies(param1, param2, topic, k=how_many)
        except Exception as e:
            print(f"Error getting analogies for {param1}, {param2}, {topic}: {e}")
            return [(0.99, topic)]    
        predicted_content = []
        for score, word in results:
            word = word.lower()
            if word.endswith('.'):  
                word = word[:-1]
            if word not in [param1.lower(), param2.lower(), topic.lower()] and word not in predicted_content:
                predicted_content.append((score, word))

        return predicted_content if predicted_content else [(0.99, topic)] # if all the predicted contents are removed, we predict the topic again
    
    def __get_topic_embedding(self, topic_word: str):
        try:
            return self.fasttext_model.get_word_vector(topic_word)
        except Exception as e:
            print(f"Exception raised in self.fasttext_model.get_word_vector(): {e}")
            return None

    def update_embedding_global(self, user_id: int, topic_word: str, is_feedback: bool = False):
        user = get_user_global(user_id) # returns None if user_id not present id db

        topic_embedding = self.__get_topic_embedding(topic_word)
        if topic_embedding is None:
            print("Failed to embed topic", flush=True)
            return None

        if user is None:
            new_user = {
                "user_id": user_id,
                "user_embedding": topic_embedding.tolist(),
                "topic_embedding_count": 1,
                "last_update": datetime.now()
            }
            create_user_profiling_global(UserProfilingGlobalModel(**new_user))
            print(f"New user with user_id: {user_id} created", flush=True)
        else:
            idd = str(user['_id'])
            del user['_id'] # we will create a new `UserProfilingGlobal`, and to do so, we need to delete the current `_id`
            # we need this to convert list to np.array to be able to make vector operations
            user['user_embedding'] = np.fromiter(user['user_embedding'], dtype=np.ndarray)
            user['user_embedding'] += (1 / user['topic_embedding_count']) * (topic_embedding - user['user_embedding'])
            if not is_feedback:
                user['topic_embedding_count'] += 1
            # convert back to list becuase mongo cannot store np.array
            user['user_embedding'] = user['user_embedding'].tolist()
            user["last_update"] = datetime.now()
            updated_user = UserProfilingGlobalModel(**user)
            update_user_profiling_global(idd, updated_user)
            print(f"User with user_id: {user_id} has been updated", flush=True)

    def update_embedding_session(self, user_id: int, session_id: int, topic_word: str, is_feedback: bool = False):
        user = get_user_session(user_id, session_id) # returns None if user_id not present id db

        topic_embedding = self.__get_topic_embedding(topic_word)
        if topic_embedding is None:
            print("Failed to embed topic", flush=True)
            return None

        if user is None:
            new_user = {
                "user_id": user_id,
                "session_id": session_id,
                "user_embedding": topic_embedding.tolist(),
                "topic_embedding_count": 1,
                "last_update": datetime.now()
            }
            create_user_profiling_session(UserProfilingSessionModel(**new_user))
            print(f"New session user with user_id: {user_id} created", flush=True)
        else:
            idd = str(user['_id'])
            del user['_id'] # we will create a new `UserProfilingSession`, and to do so, we need to delete the current `_id`
            # we need this to convert list to np.array to be able to make vector operations
            user['user_embedding'] = np.fromiter(user['user_embedding'], dtype=np.ndarray)
            user['user_embedding'] += (1 / user['topic_embedding_count']) * (topic_embedding - user['user_embedding'])
            if not is_feedback:
                user['topic_embedding_count'] += 1
            # convert back to list becuase mongo cannot store np.array
            user['user_embedding'] = user['user_embedding'].tolist()
            user['last_update'] = datetime.now()
            updated_user = UserProfilingSessionModel(**user)
            update_user_profiling_session(idd, updated_user)
            print(f"User with user_id: {user_id} has been updated", flush=True)

    def update_topic_count(self, user_id: int, topic_word: str):
        user = get_user_topic_count_global(user_id) # returns None if user_id not present id db
        # we use `new_topic` both when an existing user visit a new topic and when a new user has been created 
        new_topic = {
            "topic_name": topic_word,
            "visit_occurrences": 1,
            "last_visit": datetime.now()
        }
        if user:
            idd = str(user['_id'])
            del user['_id'] # we will create a new `UserProfilerTopicCountGlobal`, and to do so, we need to delete the current `_id`
            # we need this to convert list to np.array to be able to make vector operations
            topic_found = False
            for user_history in user['user_history']:
                if user_history['topic_name'] == topic_word:
                    user_history['visit_occurrences'] += 1
                    user_history['last_visit'] = datetime.now()
                    topic_found = True
            if not topic_found:
                user['user_history'].append(new_topic)
            
            updated_user = UserProfilingTopicCountGlobalModel(**user)
            update_user_profiling_topic_count_global(idd, updated_user)
            print(f"Topic count for user with user_id: {user_id} has been updated", flush=True)
        else:
            new_user = {
                "user_id": user_id,
                "user_history": [new_topic],
                "last_update": datetime.now()
            }
            create_user_profiling_topic_count_global(UserProfilingTopicCountGlobalModel(**new_user))
            print(f"New topic count user with user_id: {user_id} created", flush=True)



