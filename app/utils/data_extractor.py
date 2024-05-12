from bs4 import BeautifulSoup
import requests
import re
from typing import Optional
from urllib.parse import urlparse, unquote
import langid
import urllib.parse
import urllib.request
import json


class DataExtractor:
    def __init__(self, local_config) -> None:
        self.local_config = local_config
    
    def extract_all_metadata(self, html_body):
        result = {"page_description": None, "title_page": None, "page_keywords": None, "page_author": None, "page_lang": None}
        try:
            soup = BeautifulSoup(html_body, 'html.parser')
        except Exception as e:
            print(f"Exception in BeautifulSoup(): {e}", flush=True)
            return result
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name', '').lower() == 'description':
                result["page_description"] = tag.get('content', '') 
            elif tag.get('name', '').lower() == 'keywords':
                keywords = tag.get('content', '').split(',')
                if not keywords or keywords == ['']:
                    result["page_keywords"] = None
                else:
                    result["page_keywords"] = [kw.strip() for kw in keywords]
            elif tag.get('name', '').lower() == 'author':
                result["page_author"] = tag.get('content', '')
        # extract page language
        try:
            lang, _ = langid.classify(html_body)
            result["page_lang"] = lang
        except Exception as e:
            print(f"Exception in langid.classify(): {e}", flush=True)
        # extract title page
        if (title_page := soup.title) is not None:
            result['title_page'] = title_page.text.strip()
        return result
        
    def __get_relevant_link(self, page_id: str, base_url: str) -> Optional[str]:
        params = {
            'action': 'query',
            'format': 'json',
            'pageids': page_id,
            'prop': 'revisions',
            'rvprop': 'content',
            #'rvsection': 0  # First section only
        }
        try:
            response = requests.get(base_url,
                                    params=params,
                                    allow_redirects=True,
                                    headers=self.local_config["DataExtractor"]["get_request_header"],
                                    timeout=self.local_config["DataExtractor"]["get_request_timeout"])
        except Exception as e:
            print(f"Exception raised in requests.get(): {e}", flush=True)
            return None

        if response.status_code != 200:
            return None

        try:
            data = response.json()
        except Exception as e:
            print(f"Exception raised in response.json(): {e}", flush=True)
            return None

        if 'query' not in data:
            return None
        elif 'pages' not in data['query']:
            return None  # 'query' or 'pages' key is missing in the response

        content = data['query']['pages'][page_id]['revisions'][0]['*']
        links = re.findall(r'\[\[(.*?)\]\]', content)
        if links != []:
            return links[0].split('|')[0] # Return the first part if there's a display text
        else:
            return None
    
    def __extract_text_and_url_from_wikipedia(self, word):
        base_url = self.local_config["DataExtractor"]["base_wikipedia_url"]
        params = {
            'action': 'query',
            'format': 'json',
            'titles': word,
            'prop': 'extracts|pageprops|revisions|info',
            'inprop': 'url',
            #'exintro': True,
            'explaintext': True,
            'redirects': 1,
            'rvprop': 'content',
            'rvsection': 0
        }
        try:
            response = requests.get(base_url,
                                    params=params,
                                    allow_redirects=True,
                                    headers=self.local_config["DataExtractor"]["get_request_header"],
                                    timeout=self.local_config["DataExtractor"]["get_request_timeout"])
        except Exception as e:
            print(f"Exception raised in requests.get(): {e}", flush=True)
            return None

        if response.status_code != 200:
            return None

        try:
            data = response.json()
        except Exception as e:
            print(f"Exception raised in response.json(): {e}", flush=True)
            return None

        if 'query' not in data:
            return None
        elif 'pages' not in data['query']:
            return None  # 'query' or 'pages' key is missing in the response
        page_id = list(data['query']['pages'].keys())[0]
        if page_id == "-1":
            return None

        page_data = data['query']['pages'][page_id]
        if 'disambiguation' in page_data.get('pageprops', {}):
            relevant_link_title = self.__get_relevant_link(page_id, base_url)
            if relevant_link_title is not None:
                # Update params to fetch the relevant linked page
                params['titles'] = relevant_link_title

                try:
                    response = requests.get(base_url,
                                            params=params,
                                            allow_redirects=True,
                                            headers=self.local_config["DataExtractor"]["get_request_header"],
                                            timeout=self.local_config["DataExtractor"]["get_request_timeout"])
                except Exception as e:
                    print(f"Exception raised in requests.get(): {e}", flush=True)
                    return None

                if response.status_code != 200:
                    return None

                try:
                    data = response.json()
                except Exception as e:
                    print(f"Exception raised in response.json(): {e}", flush=True)
                    return None

                if 'query' not in data:
                    return None
                elif 'pages' not in data['query']:
                    return None  # 'query' or 'pages' key is missing in the response
                page_id = list(data['query']['pages'].keys())[0]
                if page_id == "-1":
                    return None
                page_data = data['query']['pages'][page_id]

        if 'extract' not in page_data:
            return None
        if page_data['extract'] == "":
            return None

        canonicalurl = ''
        if 'canonicalurl' in page_data:
            canonicalurl = page_data['canonicalurl']
        idx = page_data['extract'][:self.local_config["DataExtractor"]["max_text_len"]].rfind(".")
        page_data['extract'] = page_data['extract'][:idx] + "."

        return {'wiki_text': page_data['extract'], 'wiki_url': unquote(canonicalurl)} # TODO: this `canonicalurl` is actually the same of variable `wikipedia_link` in extract_encyclopedic_content_proposition() function

    def __extract_text(self, html_body):
        soup = BeautifulSoup(html_body, features='html.parser')
        # section, span, li, h3: maybe not unwanted
        for script in soup(["script", "style", 'iframe', 'header', 'footer', 'figure', 'link', 'a', 'img', 'svg', 'aside', 'button', 'nav']):
            script.extract()
        body_text = soup.get_text(separator=' ')
        body_text = body_text.replace('\n', ' ')
        body_text = re.sub('\s+', ' ', body_text)
        return body_text

    def get_domain(self, suggested_websites):
        weblist = []
        if not suggested_websites:
            weblist.append({"website_name": "", "url": "", "topic": "", "liked": False})
        else:
            for title, url in suggested_websites.items():                
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    domain_name = "Invalid URL"
                else:
                    domain_name = f"{parsed_url.scheme}://{parsed_url.netloc}"
                weblist.append({"website_name": domain_name, "url": url, "topic": title, "liked": False})
        return weblist


    def get_wikifier_link(self, text, lang="it"):
        # Prepare the URL.
        data = urllib.parse.urlencode([
            ("text", text), ("lang", lang),
            ("userKey", self.local_config["wikifier_key"]), 
            ("pageRankSqThreshold", "%g" % self.local_config["wikifier_threshold"]), ("applyPageRankSqThreshold", "true"),
            ("nTopDfValuesToIgnore", "3"), ("nWordsToIgnoreFromList", "3"),
            ("wikiDataClasses", "false"), ("wikiDataClassIds", "false"),
            ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "5"),
            ("includeCosines", "true"), ("maxMentionEntropy", "1")
            ])
        url = "http://www.wikifier.org/annotate-article"
        
        # Call the Wikifier and read the response.
        req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))

        # Output the annotations.
        wikifier_links = []
        for annotation in response["annotations"]:
            wikifier_links.append(unquote(annotation["url"]))
        return wikifier_links

    
    def __get_wikipedia_websites_based_on_trimmed_corrected_text_or_url(self, corrected_text_or_wiki_word):
        params = {
            'key': self.local_config["google_search_api_key"],
            'cx': self.local_config["NLPAlgorithm"]["google_cse_onlywikipedia_id"],
            'q': corrected_text_or_wiki_word
        }
        
        try:
            response = requests.get(self.local_config["NLPAlgorithm"]["base_google_url"],
                                    params=params,
                                    allow_redirects=True,
                                    headers=self.local_config["DataExtractor"]["get_request_header"],
                                    timeout=self.local_config["DataExtractor"]["get_request_timeout"])
        except Exception as e:
            print(f"Exception in request.get(): {e}", flush=True)
            return None
        if response.status_code != 200:
            print(f"Get request failed with status_code {response.status_code}", flush=True)
            return None
        data = response.json()
        items = data.get("items", None)
        if not items:
            return None
        links = [unquote(item.get("link")) for item in items if corrected_text_or_wiki_word != unquote(item.get("link").split(",")[-1])] # this IF serves to eliminate wikipedia links equal to the input
        print(f"LINK PD: {links}", flush=True)
        return links

    # if `starting_url` is a wikipedia link, we get its word and return wikipedia links based on the word.
    # if `starting_url` is NOT a wikipedia link, we take all the extracted text (`corrected_text`), remove the non alphanumeric chars and take first 5 words. Then, we get the wikipedia links based on these 5 words.
    # if the upper method does not return any wikipedia links, we take the extracted text (`corrected_text`) and brutally trim the text at 25 chars.
    def get_wikipedia_websites_based_on_context(self, starting_url, corrected_text):
        if "it.wikipedia.org/wiki/" in starting_url: # it means we are trying to get encyclopeding content starting from a wikipedia url
            print(f"corrected_text: {corrected_text}", flush=True)
            corrected_text_or_wiki_word = starting_url.split("/")[-1] # in this case, we pass Google only the wikipedia word. Otherwise we search on google using the context `corrected_text`
            print(f"corrected_text_or_url should be a word: {corrected_text_or_wiki_word}", flush=True)
        else:
            corrected_text_or_wiki_word = re.sub(r'[\W_]+', ' ', corrected_text, flags=re.UNICODE)
            corrected_text_or_wiki_word = " ".join(corrected_text_or_wiki_word.split(" ")[:self.local_config["NLPAlgorithm"]["words_to_consider_onlywikipedia"]])

        wikipedia_links = self.__get_wikipedia_websites_based_on_trimmed_corrected_text_or_url(corrected_text_or_wiki_word)
        if wikipedia_links:
            print(f"Return wikipedia links based on 5 words", flush=True)
            return wikipedia_links
        else:
            print(f"Return wikipedia links based on 25 chars", flush=True)
            wikipedia_links_25_chars_trim = corrected_text_or_wiki_word[:self.local_config["NLPAlgorithm"]["chars_to_consider_onlywikipedia"]]
            return self.__get_wikipedia_websites_based_on_trimmed_corrected_text_or_url(wikipedia_links_25_chars_trim)

    def extract_encyclopedic_content_proposition(self, starting_url, corrected_text, all_metadata, is_webmaster):
        wiki_text_and_url = None
        topic_ner = None
        list_of_links_to_consider = self.get_wikipedia_websites_based_on_context(starting_url, corrected_text)
        if not list_of_links_to_consider:
            return None, None
        for wikipedia_link in list_of_links_to_consider:
            print(f"LINK: {wikipedia_link}", flush=True)
            if wikipedia_link.split('/')[-2] != "wiki": # handling case "https://it.wikipedia.org/wiki/Persone_di_nome_Antonio/Giornalisti"
                continue
            if wikipedia_link.split('/')[-1] != starting_url.split('/')[-1] and ":" not in wikipedia_link.split('/')[-1]:
                print(f"WIKIPEDIA LINK CHOSEN: {wikipedia_link}", flush=True)
                topic_ner = wikipedia_link.split('/')[-1]
                wiki_text_and_url = self.__extract_text_and_url_from_wikipedia(topic_ner)
                break
        if not is_webmaster:
            del all_metadata["page_author"]
            del all_metadata["page_lang"]
        if wiki_text_and_url:
            wiki_text_and_url.update(all_metadata)
        return wiki_text_and_url, topic_ner

    def extract_text_and_metadata(self, url):
        text = None
        all_metadata = {"page_description": None, "title_page": None, "page_keywords": None, "page_author": None, "page_lang": None}
        if "wiki" in url:
            word = url.rsplit("/", 1)[-1]
            wiki_text_and_url = self.__extract_text_and_url_from_wikipedia(word)
            if wiki_text_and_url is None:
                return None
            text = wiki_text_and_url['wiki_text'] # in this case, `all_metadata` remain the default dict
        else:
            try:
                response = requests.get(url,
                                        allow_redirects=True,
                                        headers=self.local_config["DataExtractor"]["get_request_header"],
                                        timeout=self.local_config["DataExtractor"]["get_request_timeout"])
                print(f"RESPONSE extract_text_and_metadata: {response}", flush=True)
            except Exception as e:
                print(f"Exception in request.get(): {e}", flush=True)
                return None
            print(f"response.status_code = {response.status_code}", flush=True)
            if response.status_code == 200:
                text = self.__extract_text(response.text)
                if text is None or len(text) < self.local_config["DataExtractor"]["min_text_len"]:
                    print("Failed to extract text or text too short", flush=True)
                    return None
                idx = text[:self.local_config["DataExtractor"]["max_text_len"]].rfind(".")
                text = text[:idx]
                all_metadata = self.extract_all_metadata(response.text)
        if text is None:
            print("Failed to extract text", flush=True)
            return None
        return {'text': text, 'all_metadata': all_metadata}