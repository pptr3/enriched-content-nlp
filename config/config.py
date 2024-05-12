import json

def get_global_config():
    with open('/app2/config/global_config.json', 'r') as file_global_conf:
        global_config = json.load(file_global_conf)
    return global_config

def get_local_config():
    with open('/app2/config/local_config.json', 'r') as file_local_conf:
        local_config = json.load(file_local_conf)
    return local_config
