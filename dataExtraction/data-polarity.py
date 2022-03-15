from mongoDB import MongoDB
import pandas as pd
from textblob import TextBlob
import json
#from distutils.log import error
import logging



class Polarity():
    
    def __init__(self):
        self.db = MongoDB()
        logging.info('connected to database')


    def getSubjectivity(self,text):
        return TextBlob(text).sentiment.subjectivity    

    def getPolarity(self,text):
        return TextBlob(text).sentiment.polarity 

    def getAnalysis(self,score):
        if score < 0:
            return 'negative'

        elif score == 0:
            return 'neutral'  

        else:
            return 'positive'



    def calculatePolarity(self):
        collection = self.db.getTweetStgCollection()
        documents = pd.DataFrame(list(collection.find({})))
        logging.info('tweets fetched for polarity')


        documents['subjectivity'] = documents[['text']].applymap(
            lambda x: self.getSubjectivity(x)) 

        documents['polarityScore'] = documents[['text']].applymap(
            lambda x: self.getPolarity(x)) 

        documents['polarity'] = documents[['polarityScore']].applymap(
            lambda x: self.getAnalysis(x)
        )   

        logging.info('tweets labelled for polarity')


        records = json.loads(documents[['text','subjectivity','polarityScore','polarity']].T.to_json()).values()

        for  record in records:
            try:
                self.db.getTweetStgCollection().update({'text': record['text']},
                        {'$set': {'subjectivity': record['subjectivity'],
                                    'polarityScore': record['polarityScore'],
                                    'polarity': record['polarity']}})
                logging.info('Updation Done')
                    
            except Exception as e:
                logging.info("Update Failed")
                logging.info(e)
    

p = Polarity()
p.calculatePolarity()        