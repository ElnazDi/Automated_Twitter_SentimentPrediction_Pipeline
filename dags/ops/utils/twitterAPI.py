import os
import sys
#sys.path.append('/opt/airflow')

import twitter
import configparser
import logging
from datetime import date

class TwitterAPI():
    def __init__(self) -> None:
        logFile = f'../../logs/twitterAPIlog-{date.today()}.log'
        logging.basicConfig(filename=logFile, level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s')
        logging.info('Loading Twitter API config')
        self._loadTwitterAPI()

    def _loadConfig(self):
        ''' Load configuration variables for Tweeter API to extract Tweets'''
        logging.info('Reading config file')
        config = configparser.ConfigParser()
        config.read('/opt/airflow/config/config.ini')
        self.consumer_key = config['twitter']['api_key']
        self.consumer_secret = config['twitter']['api_key_secret']
        self.access_token = config['twitter']['access_token'] 
        self.access_token_secret = config['twitter']['access_token_secret'] 
    
    def _loadTwitterAPI(self):
        ''' Loads a Twitter API connection for retrieving tweets'''
        self._loadConfig()
        logging.info('Setting restful variables')
        self.rest_auth = twitter.oauth.OAuth(self.access_token, self.access_token_secret, self.consumer_key, self.consumer_secret)
        self.rest_api = twitter.Twitter(auth=self.rest_auth)
    
    def getTwitterRestAPI(self):
        return self.rest_api
    