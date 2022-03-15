from datetime import timedelta
from textwrap import dedent
import os
import sys
sys.path.append('/opt/airflow/dags')

from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from ops.cleaning.cleaningTasks import cleanTweets
from ops.extraction.extractionTasks import extractTweets
from ops.prediction.predictionTasks import predictTweets

# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

dag = DAG(
    'data-pipeline',
    default_args=default_args,
    description='Data Pipeline for Sentiment Analysis',
    #schedule_interval=timedelta(days=1),# Enable for automate schedule
    start_date=days_ago(0),
    tags=['dataExtraction', 'dataCleaning']
)

start_operator = DummyOperator(task_id='Begin_execution',dag=dag)
task1 = extractTweets(task_id='extract_tweets',dag=dag)
task2 = cleanTweets(task_id='data_cleaning',dag=dag)
task3 = predictTweets(task_id='emotion_prediction',dag=dag)
end_operator = DummyOperator(task_id='Stop_execution',dag=dag)

# Tasks Dependency (graphs)
start_operator >> task1 >> task2 >> task3 >> end_operator
