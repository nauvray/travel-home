from travel_home.ml_logic import registry
from travel_home.ml_logic import model as md
import os
from travel_home.params import *
from travel_home.ml_logic import utils


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

def predict(img_path : str):
    model = registry.load_travel_home_model()
    assert model is not None

    md.predict(model, img_path, load_class_names())

if __name__ == '__main__':
    print("=====TRAINING======")
    train(num_epochs = 5, force_train=True)

    img_to_predict_path = "../../00-data/seychelles.jpg"
    print("====PREDICTION=====")
    predict(img_to_predict_path)
