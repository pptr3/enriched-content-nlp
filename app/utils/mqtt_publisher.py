import paho.mqtt.client as mqtt
import json
from fastapi.responses import JSONResponse


class MqttPublisher:
    def __init__(self, local_config) -> None:
        self.local_config = local_config

    def publish_navigation_request_completion(self, user_id, session_id, url, broker_address, port, topic, requested_at, elaborated_at):
        request_completion = {
                "user_id": user_id,
                "session_id": session_id,
                "url": url,
                "requested_at": str(requested_at),
                "elaborated_at": str(elaborated_at)
        }
        if self.publish_content(broker_address, port, topic, request_completion) is True:
            print("navigation_request_completion succesfully published", flush=True)
        else:
            print("navigation_request_completion not published", flush=True)



    def publish_execution_timings_kpi(self, user_id, broker_address, port, topic, res):
        # get the execution timings
        exec_timings = {
                'user_id' : user_id,
                'NLP_EXECUTION': round(res["timing"]["time_full_pipeline"], 2),
                'PRE_PROCESSING_EXECUTION': round(res["timing"]["time_data_extraction"], 2),
                'TEXT_PROCESSING_EXECUTION_TIME': round(res["timing"]["time_text_processing"],2),
                'CONTENT_PROPOSITION_EXECUTION_TIME': round(res["timing"]["time_content_proposition"], 2),
                'CONTENT_PROCESSING_EXECUTION_TIME': round(res["timing"]["time_predicted_content"], 2)
        }
        if self.publish_content(broker_address, port, topic, exec_timings) is True:
            print("Execution timings succesfully published", flush=True)
        else:
            print("Execution timings not published", flush=True)
        

    def publish_keywords_kpi(self, user_id, broker_address, port, topic, res):
        result_dict = {}
        keywords = res["content_proposition"]["keywords"]
        print(f"keywords = {keywords}", flush=True)
        for _, (key, score, bool) in enumerate(keywords):
            result_dict[key] = {"score": score, "is_rewarded": bool}
        keywords_and_timing = {
            'user_id' : user_id,
            'keywords': result_dict,
            'keywords_extraction_time': round(res['timing']['time_extract_keywords'], 2)
        }
        if self.publish_content(broker_address, port, topic, keywords_and_timing) is True:
            print("Keywords succesfully published", flush=True)
        else:
            print("Keywords not published", flush=True)
        
    def publish_topic_kpi(self, user_id, broker_address, port, topic, res):
        data = {
            "user_id" : user_id,
            "topics": {
                res["content_proposition"]["topic"][0]: res["content_proposition"]["topic"][1]
            }, 
            "topics_extraction_time": round(res["timing"]["time_extract_topic"], 2)
        }
        if self.publish_content(broker_address, port, topic, data) is True:
            print("Topic succesfully published", flush=True)
        else:
            print("Topic not published", flush=True)
    
    def publish_ners_kpi(self, user_id, broker_address, port, topic, res):
        ners = res["content_proposition"]["ner"]
        entities = {}
        for ner in ners:
            entities[ner["word"]] = {"score": round(ner["score"], 2), "ner_class": ner["entity_group"]}

        data = {
            "user_id" : user_id,
            "entities" : entities, 
            "ners_extraction_time": round(res["timing"]["time_ner"], 2)
        }
        if self.publish_content(broker_address, port, topic, data) is True:
            print("Ner succesfully published", flush=True)
        else:
            print("Ner not published", flush=True)
        

    def publish_content(self, broker_address, port, topic, data):
        client = mqtt.Client()
        try:
            client.connect(broker_address, port)
        except Exception as e:
            print(f"Failed to connect to MQTT broker at {broker_address}:{port}, error: {e}", flush=True)
            return False
        try:
            message = json.dumps(data)
            result = client.publish(topic, message)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"Failed to publish message to topic {topic}, result code: {result.rc}")
                return False
        except Exception as e:
            print(f"Error publishing message to topic {topic}, error: {e}", flush=True)
            return False
        finally:
            try:
                client.disconnect()
            except Exception as e:
                print(f"Failed to disconnect MQTT client, error: {e}", flush=True)

        return True 

    def publish_empty(self, user_id, session_id, request_id, url, empty_encyclopedic_content):
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "url": url
        }

        # Preparing the data_encyclopedic_content dictionary
        data_encyclopedic_content = {
            "encyclopedic_content": {
                "content": empty_encyclopedic_content,
                "topic": "",
                "liked": False
            }
        }
        data_encyclopedic_content.update(data)
        self.publish_content(self.local_config["publisher_ip"], self.local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/encyclopedic_content", data_encyclopedic_content)

        # publishing FAQs
        data_faq = {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": "", "liked": False}}
        data_faq.update(data)
        self.publish_content(self.local_config["publisher_ip"], self.local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/faq", data_faq)

        # publishing website_content
        data_website_content = {
            "website_content": {
                "content": [{"website_name": "", "url": "", "topic": "", "liked": False}]
            }
        }
        data_website_content.update(data)
        self.publish_content(self.local_config["publisher_ip"], self.local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/website_content", data_website_content)

        #publishing generative_content
        data_generative_content = {
            "generative_content": {"content": "", "topic": "", "liked": False}

        } 
        data_generative_content.update(data)
        self.publish_content(self.local_config["publisher_ip"], self.local_config["publisher_port_enriched_content"], f"/nlp-response/{user_id}/generative_content", data_generative_content)

        print("All empty publish done", flush=True)
        return False
    
    def publish_empty_webmaster(self, webpage_url, broker_address, port, topic):
        res = {	
            "webpage_url" : webpage_url,
            "encyclopedic_content" : "",
            "generative_content" : "",
            "website_content" : [],
            "faqs" : [],
            
            "keywords": [],
            "topics": [],
            "ners": [],
            
            "corrige_corrections": [],

            "webpage_text" : "",
            "word_count" : 0,

            "title_page" : "", 
            "page_description": "",
            "page_keywords": [],
            "page_author": "",
            "page_lang": ""
        }
        self.publish_content(broker_address, port, topic, res)
        return False
