from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# update this line according to the path
# try to respect the path proposed
# from travel-home.dl_logic.registry import load_model

app = FastAPI()
app.state.model = load_model()

# optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # allows all methods
    allow_headers=["*"],  # allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")

def predict(X_pred:np.array) -> int:

    """
    Offers different solutions on landscape similar to what the user
    input as image
    Input is then an image converted in hex

    It returns 3 elements :
        - An image (or coordinates of a square using s2sphere)
        - A sample of image of the square
        - Text indicating the region to visit
    """

    # import the preprocess
    # from travel-home.dl_logic.preprocess import preprocess

    X_pred =                                # Image_input
    model = load_model()
    X_preproc = preprocess(X_pred)
    y_pred = app.state.model.predict(X_preproc)

    return y_pred
