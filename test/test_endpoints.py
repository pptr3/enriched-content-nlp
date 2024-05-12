import pytest
from worker import enriched_content_proposal
from worker import nlp_service_text
from worker import webmaster_content_proposal
from app.api.routers.global_config import read_config_algorithms
from app.api.routers.global_config import read_config_features
from app.api.routers.global_config import update_global_config
from app.api.routers.local_config import read_local_config
from app.mongo_models.global_config_model import GlobalConfigModel
import ray
from fastapi.responses import JSONResponse
from config.config import get_global_config
from app.api.routers import list_linguistic_analysis
from app.api.routers import list_enriched_content
from app.api.routers import list_webmaster_content
from app.mongo_models.enriched_content_model import EnrichedContentListModel
from app.mongo_models.linguistic_analysis_model import LinguisticAnalysisListModel
from app.mongo_models.webmaster_content_model import WebmasterContentListModel

text = """"
Annalisa Scarrone, nota semplicemente come Annalisa (Savona, 5 agosto 1985), è una cantautrice italiana.
Dopo alcune esperienze nell'ambito musicale con due gruppi, è divenuta nota come cantante solista nel 2011, partecipando alla decima edizione del talent show Amici di Maria De Filippi in cui ha ottenuto il Premio della critica, poi vinto anche l'anno dopo ad Amici Big.
Nel corso della sua carriera ha venduto oltre tre milioni di unità sul suolo nazionale, diventando l'artista italiana con più copie vendute in era FIMI. Ha ricevuto numerosi riconoscimenti, tra cui un Global Force Award ai Billboard Women in Music 2024 (premio dato per la prima volta a un'artista italiana),un MTV Europe Music Award, cinque Music Awards, un Premio Lunezia, due Premi Videoclip Italiano, due Velvet Awards e un premio Mia Martini, oltre ad essere stata proclamata vincitrice della sesta edizione dell'International Song Contest: The Global Sound 2013 e della trentaduesima edizione dell'OGAE Second Chance 2018.
È stata scelta per rappresentare l'Italia agli MTV Italia Awards 2013, all'OGAE Song Contest 2013 e all'OGAE Song Contest Second Chance 2024; ha inoltre ottenuto quattro candidature ai World Music Awards 2013 e una, nel 2015, ai Kids' Choice Awards e agli MTVStarOf2015.
Nel 2023 l'edizione italiana della rivista Forbes l'ha inserita fra le cento donne di successo in Italia.
"""

invalid_enriched_content_proposal_input = [
    ("", 0, 0, 0),
    ("invalid_url", 0, 0, 0),
    (None, 0, 0, 0),
    (0, 0, 0, 0),
    ("wikipedia.it/ndjdksjcdkkdd", 0, 0, 0),
    ("https://www.msn.com/it-it/notizie/italia/chiara-ferragni-mette-in-vendita-un-pigiama-dell-influencer-e-riceve-improperi-nemmeno-per-regalo/ar-AA1n2u9v", 0, 0, 0),
]

valid_enriched_content_proposal_input = [
    ("https://it.wikipedia.org/wiki/Annalisa_(cantante)", 0, 0, 0),
    ("https://mam-e.it/lintelligenza-artificiale-nella-moda-opportunita-o/", 0, 0, 0),
]

@pytest.mark.parametrize("url, user_id, session_id, request_id",
                         invalid_enriched_content_proposal_input)
def test_invalid_enriched_content_proposal(url,
                                           user_id,
                                           session_id,
                                           request_id):
    assert ray.get(enriched_content_proposal.remote(url, user_id, session_id, request_id)) is False

@pytest.mark.parametrize("url, user_id, session_id, request_id",
                         valid_enriched_content_proposal_input)
def test_valid_enriched_content_proposal(url,
                                         user_id,
                                         session_id,
                                         request_id):
    assert ray.get(enriched_content_proposal.remote(url, user_id, session_id, request_id)) is True

invalid_nlp_service_text_input = [
    ("", 0, 0, 0),
]

valid_nlp_service_text_input = [
    ("dlkfj", 0, 0, 0), # keyword extraction does not fail and take this nonsense as input
    (text, 0, 0, 0),
]

@pytest.mark.parametrize("text, user_id, session_id, request_id",
                         invalid_nlp_service_text_input)
def test_invalid_nlp_service_text(text,
                                  user_id,
                                  session_id,
                                  request_id):
    assert ray.get(nlp_service_text.remote(text, user_id, session_id, request_id)) is False

@pytest.mark.parametrize("text, user_id, session_id, request_id",
                         valid_nlp_service_text_input)
def test_invalid_nlp_service_text(text,
                                  user_id,
                                  session_id,
                                  request_id):
    assert ray.get(nlp_service_text.remote(text, user_id, session_id, request_id)) is True

invalid_webmaster_content_proposal_input = [
    # TODO: insert some valid url that we do not handle well
    "",
    "invalid_url",
    None,
    0,
    "https://www.msn.com/it-it/notizie/italia/chiara-ferragni-mette-in-vendita-un-pigiama-dell-influencer-e-riceve-improperi-nemmeno-per-regalo/ar-AA1n2u9v",
]

valid_webmaster_content_proposal_input = [
    "https://it.wikipedia.org/wiki/Annalisa_(cantante)",
]

@pytest.mark.parametrize("url", invalid_webmaster_content_proposal_input)
def test_invalid_webmaster_content_proposal(url):
    assert ray.get(webmaster_content_proposal.remote(url)) is False

@pytest.mark.parametrize("url", valid_webmaster_content_proposal_input)
def test_valid_webmaster_content_proposal(url):
    assert ray.get(webmaster_content_proposal.remote(url)) is True

def test_read_config_algorithms():
    assert read_config_algorithms().status_code == 200

def test_read_config_features():
    assert read_config_features().status_code == 200

def test_read_local_config():
    assert read_local_config().status_code == 200

def test_list_enriched_content():
    assert isinstance(list_enriched_content.list_proposed_content() , EnrichedContentListModel)

def test_list_linguistic_analysis():
    assert isinstance(list_linguistic_analysis.list_proposed_content() , LinguisticAnalysisListModel)

def test_list_linguistic_analysis():
    assert isinstance(list_webmaster_content.list_proposed_content() , WebmasterContentListModel)

# TODO: do not do user_delete, user_feedback, load_nlp_model, delete_nlp_model

# NOTE: update_global_config change the actual global config into a new one, probably we do not want to write a test here for that
# it suffices to know that it works

# NOTE: update_local_config change the actual local config into a new one, probably we do not want to write a test here for that
# it suffices to know that it works

