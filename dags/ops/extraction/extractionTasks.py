import sys
# Set the current folder to locate our custom classes within Docker
sys.path.append('/opt/airflow/dags/ops')
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from extraction.twitterExtractor import TwitterExtractor
import json

class extractTweets(BaseOperator):
    ui_color = '#89DA59'
    
    @apply_defaults
    def __init__(self,**kwargs):
        super().__init__(**kwargs)        

    def execute(self, context):
        message = "Extracting tweets"
        twitterExtractor = TwitterExtractor()
        # Read the list of topics within Docker        
        with open('/opt/airflow/config/topics.json', 'r') as topicsList:
            topics = json.load(topicsList)
        [twitterExtractor.collectTweets(topic) for topic in topics["tags"]]                
        twitterExtractor.closeDBConnection()
        print(message)
        return message