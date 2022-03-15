import imp
import sys
# Set the current folder to locate our custom classes within Docker
sys.path.append('/opt/airflow/dags/ops')
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from prediction.emotionPrediction import Prediction
import json

class predictTweets(BaseOperator):
    ui_color = '#89DA59'
    
    @apply_defaults
    def __init__(self,**kwargs):
        super().__init__(**kwargs)        

    def execute(self, context):
        message = "Running Predition"
        # TODO
        Prediction().loadData()
        print(message)
        return message