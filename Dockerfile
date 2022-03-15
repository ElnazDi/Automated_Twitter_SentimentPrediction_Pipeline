
FROM apache/airflow:2.2.3-python3.8

# Change to airflowuser
USER airflow 

# Copy file with dependencies
COPY requirementsAirflow.txt /home/airflow/requirements.txt

# change working directory to locate the file
WORKDIR /home/airflow

# install dependencies
RUN pip install -r requirements.txt

# got some errors after building

