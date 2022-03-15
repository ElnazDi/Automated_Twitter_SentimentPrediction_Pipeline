import sys
sys.path.append('/opt/airflow/dags/ops')
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from cleaning.dataCleaning import DataCleaning

class cleanTweets(BaseOperator):
    ui_color = '#89DA59'
    
    @apply_defaults
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        #self.dataCleaning = 
        print("Init finished! Ready to begin with the task")

    def execute(self, context):
        message = "Cleaning tweets"
        print("====== execute ======")
        DataCleaning().clean_tweets()
        print(message)
        return message