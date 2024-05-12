import re
import string
import emoji
from sentence_transformers import SentenceTransformer, util
from config.config import get_global_config
from urllib.parse import unquote

class TextCleaner:
    def __init__(self, global_config) -> None:
        self.global_config = global_config
        
    def __emoji_remover(self, input_text: str) -> str:
        clean_text = emoji.demojize(input_text)
        return clean_text

    def __url_ref_remover(self, text:str) -> str:
        pattern_url = re.compile(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''')
        pattern_riferimenti = re.compile(r'\[[^\]]+\]')
        testo_senza_url = pattern_url.sub('', text)
        testo_senza_riferimenti = pattern_riferimenti.sub('', testo_senza_url)
        return testo_senza_riferimenti

    def clean(self, text: str) -> str:
        text = self.__emoji_remover(text)
        text = self.__url_ref_remover(text)
        return text
    
    def process_url(self, url:str) -> str:
        url = unquote(url)
        # add http:// if `url` does not have
        url = url if url.startswith('http') else 'http://' + url
        # if a link is made like that https://www.url.com/example#stuff, we remove `#stuff`
        if "#" in url:
            url = url.split("#")[:-1][0]
        return url

    def are_text_similar(self, text_0, text_1, threshold):
        if self.global_config["coherence_evaluation"] is True:
            try:
                model = SentenceTransformer(f"app/models/{self.global_config['models']['keyword_extraction_algorithm']['model_name']}")
                embeddings_0 = model.encode(text_0, convert_to_tensor=True)
                embeddings_1 = model.encode(text_1, convert_to_tensor=True)
            except Exception as e:
                print(f"Exception raised in text encoding: {e}", flush=True)
                return False
            cosine_scores = util.cos_sim(embeddings_0, embeddings_1)
            return (cosine_scores > threshold).item()
        else:
            return False