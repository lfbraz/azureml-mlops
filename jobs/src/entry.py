import json
import joblib
import numpy as np
import pandas as pd
import os
import xgboost as xgb

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pickle')
    print(model_path)
    model = joblib.load(model_path)

# Called when a request is received
def run(data):
    data = pd.read_json(data, orient = 'split')
    data_xgb = xgb.DMatrix(data)

    # Return the prediction
    prediction = predict(data_xgb)
    return prediction

def predict(data):
    prediction = model.predict(data)[0]
    return {"churn-prediction": str(int(prediction))}
