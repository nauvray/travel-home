from fastapi import FastAPI, UploadFile, File
from travel_home.ml_logic.registry import load_travel_home_model
from travel_home.ml_logic import utils
from travel_home.ml_logic import model as md
from travel_home.params import *
from PIL import Image
import validators
import cv2
import numpy as np

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
        img = utils.get_image_from_url(image)
    else:
        img = Image.open(image)

    y_pred = md.predict(model, img, load_class_names())
    return y_pred

@app.post("/predictcustom")
async def predict(image:UploadFile=File(...)):
    # get stored model
    model = app.state.model
    assert model is not None

    img = await image.read()
    img = np.frombuffer(img,dtype=np.uint8)
    img_arr = cv2.imdecode(img,cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(img_arr,mode='RGB')

    y_pred = md.predict(model, img, load_class_names())
    return y_pred


@app.get("/")
def root():
    return {'greeting': 'Hello'}
