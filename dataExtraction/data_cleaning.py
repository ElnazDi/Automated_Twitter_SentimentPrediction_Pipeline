import logging
from mongoDB import MongoDB
import pandas as pd
from distutils.log import error
import json
import re


# class for cleaning the tweets
class DataCleaning():

    def __init__(self):
        self.db = MongoDB()
        logging.info('database connected')

    def data_preprocessing(self, text):
        # Remove any hyperlinks
        sentence = re.sub(r'https?:\/\/\S+', '', text)
        # Removing the RT
        sentence = re.sub(r'RT[\s]+', '', sentence)
        # Remove any '#'
        sentence = re.sub(r'#', '', sentence)
        # Remove the '\n' string
        sentence = re.sub('\\n', ' ', sentence)
        # Removing the @mention
        sentence = re.sub(r'@[A-Za-z0-9]+', '', sentence)
        # Data Cleansing
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Removing numbers
        sentence = re.sub(r'[0-9]', '', sentence)

        return sentence

    def clean_tweets(self):
        collection = self.db.getTweetCollection()
        print('collection', collection)
        documents = pd.DataFrame(list(collection.find({})))
        logging.info('tweets fetched for cleaning')


        documents['text'] = documents[['text']].applymap(
            lambda x: self.data_preprocessing(x))
        logging.info('tweets preprocessed')


        documents = documents.drop_duplicates(subset=['text'], keep='first')
        logging.info('dropped duplicates')

    
        records = json.loads(documents[['text']].T.to_json()).values()
        try:
            self.db.getTweetStgCollection().insert_many(records)
            logging.info('Insertion Done')
        except error as e:
            logging.info("insertion Failed")
            logging.info(e)


dc = DataCleaning()
dc.clean_tweets()
