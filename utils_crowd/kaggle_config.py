"""
This file contains the Kaggle API credentials.
"""

import json

try:
    with open('kaggle.json', 'r') as file:
        data = json.load(file)
except:
    print("The kaggle.json file is not found in the root directory, please check the kaggle.json file")


KAGGLE_USERNAME = data['username']
KAGGLE_KEY = data['key']