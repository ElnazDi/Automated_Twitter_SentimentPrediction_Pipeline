# Class for Fetching Tweets from API: Containing many information 
# After cleaning and preprocessing, these tweets will be used for our prediction purpose to lable them with different emotions class
# LSTM Model is saved in the Mlflow
import os
import sys
sys.path.append('/opt/airflow/dags/ops')

from pprint import pprint
from utils.mongoDB import MongoDB
from utils.twitterAPI import TwitterAPI
import logging
from datetime import date


class TwitterExtractor():
    '''Pure Tweets extractor with all the informations appended to each Tweet'''

    def __init__(self):
        '''Class Attributes: Mongodb, Tweeter API, Limited to English Tweets and number of them'''
        self.db = MongoDB()
        self.twitterapi = TwitterAPI()
        self.language = "en"
        self.default_tweets = 50

        logFile = f'twitterExtraction-{date.today()}.log'
        logging.basicConfig(filename=logFile, level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s')
        logging.info('Configuration for local development')

    
    # TODO logic
    def collectTweets(self, q):
        ''' Class Method: Search for specific topics, desired language and number of tweets'''
        logging.info('Searching for tweets')
        #q = "MLOps"                              
        search_results = self.twitterapi.getTwitterRestAPI().search.tweets(count=self.default_tweets,q=q, lang=self.language) #you can use both q and geocode
        statuses = search_results["statuses"]
        #since_id_new = statuses[-1]['id']
        for status in statuses:
            try:
                self.db.getTweetCollection().insert_one(status)
                pprint(status['created_at']) # print the date of the collected tweets
                logging.info(status['created_at']) # print the date of the collected tweets
            except Exception as e: 
                logging.error(f'Tweets extraction failed')
                print(e)
                logging.error(e)
        logging.info('Finishing reading tweets')

    def collectOldTweets(self, q):
        '''Class Method: inserting old tweets up to past 7 days'''
        since_id_old = 0
        while(since_id_new != since_id_old):
            since_id_old = since_id_new
            search_results = self.twitterapi.getTwitterRestAPI().search.tweets( count=self.default_tweets,q=q,
                                lang=self.language, max_id= since_id_new)
            statuses = search_results["statuses"]
            since_id_new = statuses[-1]['id']
            for statuse in statuses:
                try:
                    self.db.getTweetCollection().insert_one(statuse)
                    pprint(statuse['created_at']) # print the date of the collected tweets
                    logging.info(statuse['created_at']) # print the date of the collected tweets
                except Exception as e:
                    logging.error(f'Tweets extraction failed')
                    print(e)
                    logging.error(e)
    
    def closeDBConnection(self):
        '''Class Method: Close DB connection'''
        self.db.closeConnection()

