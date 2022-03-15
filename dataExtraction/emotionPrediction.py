# prediction
import mlflow
import pandas as pd


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    modelName = 'EmotionsModel'
    # When you are happy with the model change to Production (UI and here)
    stage = 'Staging'

    model = mlflow.pyfunc.load_model(
        model_uri=f'models:/{modelName}/{stage}'
    )

    # should replace with the cleaned data from Twitter API
    # input = pd.read_csv("to_predict.csv", sep=";")
    result = model.predict(input)
    print(result)
