from travel_home.ml_logic import registry
from travel_home.ml_logic import model as md
import os
from travel_home.params import *
import validators
from PIL import Image
import requests


def train(num_epochs : int, force_train : bool):
    # add "npy" folder to data_dir as images are saved in "npy" folder in the bucket
    data_dir = os.path.join(WORKING_DIR, "npy")
    # load pre trained model if it exists
    model = registry.load_travel_home_model()

    if ((model is None) or force_train):
        if (model is None):
            print("no model found, start training...")
        else:
            print("force model training...")
        # prepare inputs
        md.prepare_train_val_folders(data_dir)
        dataloaders = md.prepare_input_train(data_dir)
        # load model
        model = md.load_model()
        # train model
        model = md.train_model(dataloaders, model, num_epochs=num_epochs)
        # Save model weights locally and in GCS
        registry.save_travel_home_model(model=model)
    else:
        print("No training required")

def predict(image_path : str):
    model = registry.load_travel_home_model()
    assert model is not None

    if (validators.url(image_path)):
        img = Image.open(requests.get(image_path, stream = True).raw)
    else:
        img = Image.open(image_path)

    md.predict(model, img, load_class_names())

if __name__ == '__main__':
    print("=====TRAINING======")
    num_epochs = 50
    train(num_epochs=num_epochs, force_train=False)

    print("====PREDICTION=====")
    image_path = "../../00-data/seychelles.jpg"
    # image_path = "https://img.traveltriangle.com/blog/wp-content/uploads/2017/10/Cover10.jpg"
    predict(image_path)
