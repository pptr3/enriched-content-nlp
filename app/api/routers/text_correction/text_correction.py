import logging
import re
import urllib.parse
import requests

from config.config import get_local_config


ping_count = '4'
instances = ["DOCKER_ON_INTEL_XEON_16GB_2RX8", "t3.medium_2vCPU_4GB", "t3.xlarge_4vCPU_16GB", "t3.2xlarge_8vCPU_32GB"]

base_url = get_local_config()["CorrigeAlgorithm"]["base_url"]
id_utente = get_local_config()["Corrige_id_utente"]
url = base_url + id_utente + "&Content="

username = get_local_config()["Corrige_username"]
password = get_local_config()["Corrige_password"]
header = get_local_config()["CorrigeAlgorithm"]["header"]
footer = get_local_config()["CorrigeAlgorithm"]["footer"]

def fix_text_with_corrige(input_text:str, ner:bool=False) -> str:
    try:
        logging.debug("Text uploaded by the user: " + input_text)
        try:
            content1 = urllib.parse.quote_plus(header + input_text + footer)
        except Exception as e:
            logging.debug(f"Exception raised in urllib.parse.quote_plus(): {e}")
            return input_text, [] # empty list means that there are no corrections made
        url1 = url + content1

        # print("API url: ", url1)
        logging.info("Making the call to Corrige...")

        try:
            response = requests.post(url1, auth=(username, password))
            logging.debug("Request done.")
        except Exception as e:
            logging.debug(f"Exception raised in requests.post(): {e}")
            return input_text, [] # empty list means that there are no corrections made

        #print("Response code: ", response.status_code)
        input_response = response.text
        #print("API response: ", input_response)

        #logging.info("it returned this output:" + str(input_response))

        err_corr_pattern = r'<SEGNALA (.*?) </SEGNALA>'

        ante_v_pattern = r'ANTE="(.*?)" POST'
        post_v_pattern = r'POST="(.*?)">'

        ante_s_pattern = r'<CCZANTE><!\[CDATA\[(.*?)\|\]\]></CCZANTE>'
        post_s_pattern = r'<CCZPOST><!\[CDATA\[(.*?)\]\]></CCZPOST>'

        wrong_word_pattern = r'Alfabetico="(.*?)" Tipo'
        correction_pattern = r'<CORRIGE IDscelta=(.*?)</CORRIGE>'

        # TODO(tommaso): does finditer ever raise an exception to user? I do not think so
        ante_v_match = re.finditer(ante_v_pattern, input_response)
        post_v_match = re.finditer(post_v_pattern, input_response)

        ante_s_match = re.finditer(ante_s_pattern, input_response)
        post_s_match = re.finditer(post_s_pattern, input_response)

        wrong_word_match = re.finditer(wrong_word_pattern, input_response)
        correction_match = re.finditer(correction_pattern, input_response)

        err_corr_match = re.finditer(err_corr_pattern, input_response)

        ante_value = []
        post_value = []

        ante_string = []
        post_string = []

        wrong_word = []
        correction_word = {}

        for match in ante_v_match:
            ante_value.append(match.group(1))

        for match in post_v_match:
            post_value.append(match.group(1))

        for match in ante_s_match:
            ante_string.append(match.group(1))

        for match in post_s_match:
            post_string.append(match.group(1))

        for match in wrong_word_match:
            wrong_word.append(re.sub(r'\d+', '', match.group(1)).lower().replace("_", "'"))

        index_word_with_correction = []
        for i, match in enumerate(err_corr_match):
            check_correction_string = match.group(1)
            if check_correction_string is not None and r'<CORRIGE' in check_correction_string:
                index_word_with_correction.append(i)
        if len(index_word_with_correction) > 0:

            #print("errors: ", wrong_word)
            correction_num = [999]

            for match in correction_match:
                word = match.group(1)
                try:
                    correction_num.append(int(re.sub(r'[^0-9]', '', word[:5])))
                except ValueError:
                    correction_num.append(1)
                word = word[word.find(">") + 1:].replace("_", " ")

                if correction_num[-1] <= correction_num[-2]:
                    correction_word[len(correction_word.keys()) + 1] = [word]
                else:
                    correction_word[len(correction_word.keys())].append(word)

            def ante_post_handler(input_text, wrong_word, ante_value, post_value):
                index = input_text.find(wrong_word[0])
                index_last_char = index + len(wrong_word)

                string_range = index - int(ante_value[0]), index + int(post_value[0])

                if string_range[1] == index:
                    wrong_word_full = input_text[index - int(ante_value[0]): index] + wrong_word
                elif string_range[0] == index:
                    wrong_word_full = wrong_word + input_text[index_last_char: index_last_char + int(post_value[0])]
                else:
                    wrong_word_full = input_text[index - int(ante_value[0]): index] + wrong_word + input_text[
                                                                                                index_last_char: index_last_char + int(
                                                                                                    post_value[0])]

                return wrong_word_full
            wrong_word_f = []
            for i in range(0, len(wrong_word)):
                if i in index_word_with_correction:
                    wrong_word_f.append(ante_post_handler(input_text, wrong_word[i], ante_value[i], post_value[i]))
            #print(wrong_word)
            #print("errors with ante/post: ", wrong_word_f)
            def erase_accented_correction(errors_pair):
                list_out = []
                if not errors_pair:
                    return []
                for key, value in errors_pair.items():
                    if '&#' not in value: # this means that the corrected word had an accent and it has been wrongly corrected by Corrige
                        list_out.append({"error" : key, "correction" : value})
                return list_out

            err_fix_pair = {key: correction_word[i + 1][0] for i, key in enumerate(wrong_word_f)}

            wrong_keys = (re.escape(k) for k in err_fix_pair.keys())
            wrong_sub_pattern = re.compile(r'\b(' + '|'.join(wrong_keys) + r')\b')
            string_result = None
            if ner:
                string_result = wrong_sub_pattern.sub(lambda x: err_fix_pair[x.group()], input_text)
            else:
                string_result = wrong_sub_pattern.sub(lambda x: err_fix_pair[x.group()], input_text.lower())
            list_dict_with_correction_without_accents = erase_accented_correction(err_fix_pair)
            logging.info("Edited text: " + string_result)
            
            return string_result, list_dict_with_correction_without_accents
        else:
            logging.info("No changes made for input text (no errors)")
            return input_text, [] # empty list means that there are no corrections made
    except Exception as e:
        print(f"Exception raised in fix_text_with_corrige: {e}", flush=True)
        return input_text, [] # empty list means that there are no corrections made
    

# print(fix_text_with_corrige("L’elaborazione del linguaggio naturale e la forza trainante dell intelligewwwwwwnza artificiale in molte moderne applicazioni del mondo reale. Ecco alcuni esempi:rilevamento dello spam: potresti non pensare al rilevamento dello spam come una soluzione NLP, ma le migliorii tecnologie di rilevamento dello spam utilizzano la classificazione del testo della NLP funzionalità per scansionare le e-mail alla ricerca di linguaggio che spesso indica spam o phishing. Questi indicatori possono includere un uso eccessivo di termini finanziari, caratteristici grammatica errata, linguaggio minaccioso, urgenza inappropriata, nomi di aziende errate e altro ancora. Il rilevamento dello spam e uno dei pochi problemi di NLP affrontati dagli esperti considera per lo più risolto (anche se potresti sostenere che questo non corrisponde alla tua esperienza con la posta elettronic L’elaborazione del linguaggio naturale e la forza trainante dell intelligewwwwwwnza artificiale in molte moderne applicazioni del mondo reale. Ecco alcuni esempi:rilevamento dello spam: potresti non pensare al rilevamento dello spam come una soluzione NLP, ma le migliorii tecnologie di rilevamento dello spam utilizzano la classificazione del testo della NLP funzionalità per scansionare le e-mail alla ricerca di linguaggio che spesso indica spam o phishing. Questi indicatori possono includere un uso eccessivo di termini finanziari, caratteristici grammatica errata, linguaggio minaccioso, urgenza inappropriata, nomi di aziende errate e altro ancora. Il rilevamento dello spam e uno dei pochi problemi di NLP affrontati dagli esperti considera per lo più risolto (anche se potresti sostenere che questo non corrisponde alla tua esperienza con la posta elettronica).", ner=True))