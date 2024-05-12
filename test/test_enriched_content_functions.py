from app.utils.data_extractor import DataExtractor
from app.utils.text_cleaner import TextCleaner
from app.utils.nlp_algorithm import UnsharedModelsActor
from config.config import get_global_config, get_local_config
import pytest
import ray

kw_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"
topic_classifier = f"app/models/{get_global_config()['models']['topic_extraction_algorithm']['model_name']}"
are_text_similar_model = f"app/models/{get_global_config()['models']['keyword_extraction_algorithm']['model_name']}"

nlp_algorithm = UnsharedModelsActor.remote(get_local_config(), get_global_config(), kw_model, topic_classifier)
text_cleaner = TextCleaner(get_global_config())
data_extractor = DataExtractor(get_local_config())

empty_ner = [{"entity_group": "", "score": 0, "word": None}]
empty_metadata = {"page_description": None, "title_page": None, "page_keywords": None, "page_author": None, "page_lang": None}
empty_faq = {"FAQ": {"content": [{"question": "", "answer": ""}], "topic": ""}}

valid_ner = [{"entity_group": "PER", "score": 0, "word": "Luciano Spalletti"}]
valid_metadata = {
    'page_description': 'Il settore internazionale della moda e del lusso può vantare "una buona divulgazione e trasparenza sulle informazioni ambientali e su gran parte delle politiche Esg richieste da Onu, Ocse e Ue". (ANSA)', 'title_page': 'Il settore della moda promosso nel rating di sostenibilità - Moda - Ansa.it',
    'page_keywords':    ['rating', 'settore', 'società', 'politica', 'divulgazione'],
    'page_author':      'Agenzia ANSA',
    'page_lang':        'it'
}

text = """"
Annalisa Scarrone, nota semplicemente come Annalisa (Savona, 5 agosto 1985), è una cantautrice italiana.
Dopo alcune esperienze nell'ambito musicale con due gruppi, è divenuta nota come cantante solista nel 2011, partecipando alla decima edizione del talent show Amici di Maria De Filippi in cui ha ottenuto il Premio della critica, poi vinto anche l'anno dopo ad Amici Big.
Nel corso della sua carriera ha venduto oltre tre milioni di unità sul suolo nazionale, diventando l'artista italiana con più copie vendute in era FIMI. Ha ricevuto numerosi riconoscimenti, tra cui un Global Force Award ai Billboard Women in Music 2024 (premio dato per la prima volta a un'artista italiana),un MTV Europe Music Award, cinque Music Awards, un Premio Lunezia, due Premi Videoclip Italiano, due Velvet Awards e un premio Mia Martini, oltre ad essere stata proclamata vincitrice della sesta edizione dell'International Song Contest: The Global Sound 2013 e della trentaduesima edizione dell'OGAE Second Chance 2018.
È stata scelta per rappresentare l'Italia agli MTV Italia Awards 2013, all'OGAE Song Contest 2013 e all'OGAE Song Contest Second Chance 2024; ha inoltre ottenuto quattro candidature ai World Music Awards 2013 e una, nel 2015, ai Kids' Choice Awards e agli MTVStarOf2015.
Nel 2023 l'edizione italiana della rivista Forbes l'ha inserita fra le cento donne di successo in Italia.
"""

# Encyclopedic content
invalid_encyclopedic_content_input = [
    (text, empty_ner.copy(), empty_metadata.copy(), False),
]

valid_encyclopedic_content_input = [
    (text, valid_ner.copy(), empty_metadata.copy(), False),
    (text, valid_ner.copy(), valid_metadata.copy(), False),
]

@pytest.mark.parametrize("text, empty_ner, empty_metadata, is_webmaster", invalid_encyclopedic_content_input)
def test_invalid_encyclopedic_content(text, empty_ner, empty_metadata, is_webmaster):
    encyclopedic_content, topic_ner, ners_without_wiki_ner = data_extractor.extract_encyclopedic_content_proposition("",
                                                                                                                     text,
                                                                                                                     empty_ner,
                                                                                                                     empty_metadata,
                                                                                                                     is_webmaster=is_webmaster)

    assert encyclopedic_content is None, "Expected encyclopedic_content to be None"
    assert topic_ner == "", "Expected topic_ner to be empty string"
    assert ners_without_wiki_ner == empty_ner, "Expected ners_without_wiki_ner to be empty_ner"

@pytest.mark.parametrize("text, valid_ner, empty_metadata, is_webmaster", valid_encyclopedic_content_input)
def test_valid_encyclopedic_content(text, valid_ner, empty_metadata, is_webmaster):
    encyclopedic_content, topic_ner, ners_without_wiki_ner = data_extractor.extract_encyclopedic_content_proposition("",
                                                                                                                     text,
                                                                                                                     valid_ner,
                                                                                                                     empty_metadata,
                                                                                                                     is_webmaster=is_webmaster)
    assert encyclopedic_content is not None, "Expected valid encyclopedic content for a valid ner input"
    assert topic_ner != "", "Expected topic_ner to be valid for a valid ner input"
    assert ners_without_wiki_ner == [], "Expected an empty list"

# FAQ
invalid_faq_input = [
    (text, 0,   5),
    (text, 300, 0),
]

valid_faq_input = [
    (text, 300, 5),
]

@pytest.mark.parametrize("text, max_tokens, how_many", invalid_faq_input)
def test_invalid_faq(text, max_tokens, how_many):
    faq = ray.get(nlp_algorithm.faq_generation.remote(text, max_tokens, how_many))
    assert faq == empty_faq

@pytest.mark.parametrize("text, max_tokens, how_many", valid_faq_input)
def test_valid_faq(text, max_tokens, how_many):
    faq = ray.get(nlp_algorithm.faq_generation.remote(text, max_tokens, how_many))
    question_answer_concatenation = ""
    for question_answer_pair in faq["FAQ"]["content"]:
        question_answer_concatenation += (question_answer_pair["question"] + " " + question_answer_pair["answer"])
    assert faq != empty_faq, "Expected some faq generated for valid input"
    assert text_cleaner.are_text_similar(text, question_answer_concatenation, 0.7) is True, "Expected semantic meaning of text similar to the generated faq"

# Generative content
invalid_generative_content_input = [
    ("brief text", "",         0),
    (text,         "",         0),
    (text,         "fortezza", 0),
]

@pytest.mark.parametrize("corrected_text, topic, max_tokens", invalid_generative_content_input)
def test_invalid_generate_proposed_content_based_on_topic(corrected_text, topic, max_tokens):
    generative_content = ray.get(nlp_algorithm.generate_proposed_content_based_on_topic.remote(corrected_text,
                                                                                               topic,
                                                                                               max_tokens))
    assert generative_content is None

valid_generative_content_input = [
    (text, "annalisa", 250), # NOTE: actually every topic should be fine, even if it's not relevant to text
]

@pytest.mark.parametrize("corrected_text, topic, max_tokens", valid_generative_content_input)
def test_valid_generate_proposed_content_based_on_topic(corrected_text, topic, max_tokens):
    generative_content = ray.get(nlp_algorithm.generate_proposed_content_based_on_topic.remote(corrected_text,
                                                                                               topic,
                                                                                               max_tokens))
    assert text_cleaner.are_text_similar(text, generative_content, 0.7) is True

# Web sites content proposition
# NOTE: predicted_content example: [(0.99, topic)]
# NOTE: predicted_content score never used always put it to 0.1
invalid_test_cases_web_sites_content = [
    ("", {'ner': None, 'keywords': ["dskljfd"]}, [(0.1, "predicted_content")], "topic", 5),
    ("", {'ner': None, 'keywords': ["ciao"]},    [(0.1, "")],                  "",      5),
    ("", {'ner': None, 'keywords': ["ab"]},      [(0.1, "")],                  "",      5),
]

@pytest.mark.parametrize("url, ner_keywords_combined_dict, predicted_content, topic, how_many", invalid_test_cases_web_sites_content)
def test_invalid_web_sites_content_proposition(url, ner_keywords_combined_dict, predicted_content, topic, how_many):
    suggested_websites = ray.get(nlp_algorithm.web_sites_content_proposition.remote(url, ner_keywords_combined_dict, predicted_content, topic, how_many))
    assert suggested_websites == {} or suggested_websites is None, "Expected a {} or None for invalid inputs." # when how_many < 1 we immediately return None

# NOTE: predicted_content example: [(0.99, topic)]
# NOTE: predicted_content score never used always put it to 0.1
test_valid_ner = {"entity_group": "LOC", "score": 0.99, "word": "Santa Maria del Monte"}
valid_test_cases_web_sites_content = [
    ("",                                                  {'ner': [test_valid_ner], 'keywords': ["test"]}, [(0.1, "music")], "song", 5),
    ("https://it.wikipedia.org/wiki/Annalisa_(cantante)", {'ner': [test_valid_ner], 'keywords': ["test"]}, [(0.1, "music")], "song", 5),
    ("htps://www.example.com/page",                       {'ner': [test_valid_ner], 'keywords': ["test"]}, [(0.1, "music")], "song", 5),
    ("some meaningless text",                             {'ner': [test_valid_ner], 'keywords': ["test"]}, [(0.1, "music")], "song", 5),
]

@pytest.mark.parametrize("url, ner_keywords_combined_dict, predicted_content, topic, how_many", valid_test_cases_web_sites_content)
def test_valid_web_sites_content_proposition(url, ner_keywords_combined_dict, predicted_content, topic, how_many):
    suggested_websites = ray.get(nlp_algorithm.web_sites_content_proposition.remote(url, ner_keywords_combined_dict, predicted_content, topic, how_many))
    assert suggested_websites != {}, "Expected websites suggestion for valid input"
