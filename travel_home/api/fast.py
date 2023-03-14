from fastapi import FastAPI
import numpy as np
from ml_logic import registry

app = FastAPI()

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(X_pred:np.array) -> int:
    # load latest model saved in gcs
    model = registry.load_travel_home_model()
    assert model is not None

    # X_processed = preprocessor.preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)
    # prediction =  y_pred.item()
    # return {"fare_amount" : prediction}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
