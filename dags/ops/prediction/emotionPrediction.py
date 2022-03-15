# prediction
import sys
sys.path.append('/opt/airflow/dags/ops')
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import logging
from utils.mongoDB import MongoDB
import json



class Prediction():
    
    def __init__(self):
        self.db = MongoDB()
        logging.info('database connected')
        self.model = load_model("/opt/airflow/dags/ops/models/my_model")
        print("model loaded")
        self.labels = ['empty', 'sadness', 'enthusiasm' ,'neutral', 'worry' ,'surprise', 'love', 'fun',
                'hate', 'happiness', 'boredom' ,'relief', 'anger']


    def preprocessTex(self, text):
        """ Apply cleansing in the text before prediction"""
        tokenizer = Tokenizer(num_words=500, split=' ')
        tokenizer.fit_on_texts([text])
        seq = tokenizer.texts_to_sequences([text])
        X = pad_sequences(seq,23)                
        return self.predictEmotion(X)

    def predictEmotion(self, sequenceTweet):
        # Change location for keras model in Airflow
        #model = load_model("my_model")  
        pred = self.model.predict(sequenceTweet)
        #print("predicted: ",pred)
        sentiment_class=self.labels[np.argmax(pred)]
        logging.info(f'Sentiment: {sentiment_class}')
        #print(sentiment_class)
        return sentiment_class


    def loadData(self):
        """Load data from Mongo DB after Tweets extraction"""
        
        collection = self.db.getTweetStgCollection()
        print('collection', collection)
        documents = pd.DataFrame(list(collection.find({})))
        logging.info(f'tweets fetched for prediction {len(documents)}')
        print(f'tweets fetched for prediction {len(documents)}')
                
        if len(documents) > 0:        
            documents["predict"] = documents[['text']].applymap(
                lambda x: self.preprocessTex(x))
            logging.info('tweets preprocessed')
            print('tweets preprocessed')
            
            records = json.loads(documents[['text','predict']].T.to_json()).values()
            print(f"Records: {len(records)}")
            try:
                self.db.getTweetFinalCollection().insert_many(records)
                logging.info('Tweets with Prediction Done')
            except Exception as e:
                logging.info("Insertion Failed")
                logging.info(e)
        else:
            print("No tweets :(")
        self.db.closeConnection()    
    