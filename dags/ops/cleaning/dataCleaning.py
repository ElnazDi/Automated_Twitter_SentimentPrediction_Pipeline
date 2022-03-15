# Class for Cleaning Tweets that are fetched through API
# Also used for Cleaning Tweets during training the Model
import os
import sys
sys.path.append('/opt/airflow/dags/ops')
from datetime import datetime, date
from utils.mongoDB import MongoDB
import pandas as pd
from distutils.log import error
import json
import re
import logging



# class for cleaning the tweets
class DataCleaning():

    def __init__(self):
        '''Connection set to Mongodb client'''
        self.db = MongoDB()
        self.start_process = datetime.now()
        logFile = f'DataCleaning-{date.today()}.log'
        logging.basicConfig(filename=logFile, level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s')
        print('MongoDB ready!')

    def data_preprocessing(self, text):
        ''' Cleaning tweets from unalphabetic charactors '''
        # Remove any hyperlinks
        sentence = re.sub(r'https?:\/\/\S+', '', text)
        # Removing the RT - ReTweets
        sentence = re.sub(r'RT[\s]+', '', sentence)
        # Remove any '#'
        sentence = re.sub(r'#', '', sentence)
        # Remove the '\n' string
        sentence = re.sub('\\n', ' ', sentence)
        # Removing the @mention
        sentence = re.sub(r'@[A-Za-z0-9]+', '', sentence)
        # Data Cleansing - Other than words ans spaces
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Removing numbers
        sentence = re.sub(r'[0-9]', '', sentence)

        return sentence

    def clean_tweets(self):
        '''Fetching Tweets from Mongodb, Cleaning, Inserting into Mongodb'''
        start_time = datetime.now()
        logging.info(f'{start_time} Starting data cleaning process')  
        logging.info('Reading collection')
        try:            
            logging.info('tweets fetched for cleaning')
            documents = pd.DataFrame(list(self.db.getTweetCollection().find({})))
            documents['text'] = documents[['text']].applymap(lambda x: self.data_preprocessing(x))
            logging.info('tweets preprocessed')
            documents = documents.drop_duplicates(subset=['text'], keep='first')
            logging.info('dropped duplicates')
            records = json.loads(documents[['text']].T.to_json()).values()
            self.db.getTweetStgCollection().insert_many(records)
            logging.info('Insertion Done')
            self.db.closeConnection()
        except Exception as e:
            logging.error(f'Problem in reading and/or inserting documents')
            print(e)
            logging.error(e)
            


#dc = DataCleaning()
#dc.clean_tweets()
