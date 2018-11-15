import json
import os
from types import SimpleNamespace

def dict_to_sns(d):
    return SimpleNamespace(**d)

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config = json.load(config_file, object_hook=dict_to_sns)

    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    return config
