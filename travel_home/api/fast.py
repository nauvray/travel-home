from fastapi import FastAPI
from ml_logic import registry
from travel_home.ml_logic import model as md
from travel_home.params import *

app = FastAPI()

@app.get("/predict")
def predict(img):
    # load latest model saved in gcs
    model = registry.load_travel_home_model()
    assert model is not None

    y_pred = md.predict(model, img, load_class_names())
    return y_pred

@app.get("/")
def root():
    return {'greeting': 'Hello'}
