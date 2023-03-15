from travel_home.ml_logic import registry
from travel_home.ml_logic import model as md
from torchvision import datasets
import os

def train(data_dir : str, num_epochs : int, force_train : bool):
    # load pre trained model if it exists
    model = registry.load_travel_home_model(data_dir)

    if ((model is None) or force_train):
        print("model training...")
        # prepare inputs
        md.prepare_train_val_folders(data_dir)
        dataloaders, image_datasets = md.prepare_input_train(data_dir)
        # load model
        model = md.load_model(image_datasets)
        # train model
        model = md.train_model(dataloaders, image_datasets, model, data_dir, num_epochs=num_epochs)
        # Save model weights locally and in GCS
        registry.save_travel_home_model(data_dir, model=model)

def predict(data_dir : str, img_path : str):
    model = registry.load_travel_home_model(data_dir)
    assert model is not None

    train_folder = datasets.DatasetFolder(os.path.join(data_dir, "train"), loader=md.npy_loader, extensions=['.npy'], transform=md.images_transformer("train"))
    md.predict(model, img_path, train_folder.classes)

if __name__ == '__main__':
    data_dir = "/mnt/disks/disk-1/model-2/"
    num_epochs = 50
    #  img_to_predict_path = "../../00-data/seychelles.jpg"

    data_dir = os.path.join(data_dir, "npy")
    train(data_dir, num_epochs, force_train=True)
    # predict(data_dir, img_to_predict_path)
