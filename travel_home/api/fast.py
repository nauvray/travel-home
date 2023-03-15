from fastapi import FastAPI
from travel_home.ml_logic.registry import load_travel_home_model
from travel_home.ml_logic import model as md
from travel_home.params import *
from PIL import Image
import validators
import requests

app = FastAPI()
app.state.model = load_travel_home_model()

# http://127.0.0.1:8000/predict?image=https://img.traveltriangle.com/blog/wp-content/uploads/2017/10/Cover10.jpg
# http://127.0.0.1:8000/predict?image=../../00-data/seychelles.jpg
@app.get("/predict")
def predict(image):
    # get stored model
    model = app.state.model
    assert model is not None

    if (validators.url(image)):
        img = Image.open(requests.get(image, stream = True).raw)
    else:
        img = Image.open(image)

    md.predict(model, img, load_class_names())

    y_pred = md.predict(model, img, load_class_names())
    return y_pred

@app.get("/")
def root():
    return {'greeting': 'Hello'}
