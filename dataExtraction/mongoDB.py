import os
import sys
#sys.path.append('/opt/airflow')

from pymongo import MongoClient
import configparser
import logging
from datetime import date


class MongoDB():
    
    def __init__(self) -> None:
        logFile = f'logs/mongoDBlog-{date.today()}.log'
        logging.basicConfig(filename=logFile, level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s')
        logging.info('Loading Mongo DB config')
        self._loadDBConnection()

    def _loadConfig(self):
        ''' Load configuration variables to extract Tweets'''
        try:
            logging.info('Reding config file')
            config = configparser.ConfigParser()
            config.read('config.ini')
            self.mongodb_connect = config['mongo']['connection']
        except:
            raise logging.exception('Problem reading the file')

    def _loadDBConnection(self):
        '''Loads a Mongo connection after setting up the necessary configuration parameters'''
        try:
            self._loadConfig()
            logging.info('Setting Mongo settings')
            self.client = MongoClient(self.mongodb_connect)
            self.db = self.client.MLOps
            self.tweet_collection = self.db.tweet_collection_v4
            self.tweet_stg_collection = self.db.tweet_stg_collection_v4
            self.tweet_final_collection = self.db.tweet_final_collection_v4

        except:
            raise logging.exception('Problem initializing db  and collection')
    
    def getTweetCollection(self):
        ''' Return Tweet collection to insert results'''
        return self.tweet_collection
    
    def getTweetStgCollection(self):
        ''' Return Tweet Staging collection to insert results'''
        return self.tweet_stg_collection # change for staging collection
    
    def getTweetFinalCollection(self):
        ''' Return Tweet Final collection with prediction'''
        return self.tweet_final_collection 
    
    def closeConnection(self):
        '''Close connection in Mongo'''
        self.client.close()
        