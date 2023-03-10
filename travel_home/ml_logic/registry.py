from google.cloud import storage
from travel_home.params import *
from travel_home.ml_logic import model
import torch
import time

def load_travel_home_model(local_registry_path):
    """
    Return a saved model from GCS (most recent one)
    Return None (but do not Raise) if no model found
    """
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))
    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_blob_name = latest_blob.name.split("/")[-1]
        latest_model_path = os.path.join(local_registry_path, latest_blob_name)
        latest_blob.download_to_filename(latest_model_path)

        # Loading the model
        resnet_model = model.load_model()
        checkpoint = torch.load(latest_model_path)
        resnet_model.load_state_dict(checkpoint)

        print(f"ā Latest model {latest_blob_name} downloaded from cloud storage")
        return resnet_model
    except:
        print(f"\nā No model found on GCS bucket {BUCKET_NAME}")
        return None

def save_travel_home_model(local_registry_path, model):
    # Saving the model locally
    # save the state of the model (i.e. the weights)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_model_path = os.path.join(local_registry_path, f"{timestamp}.pth")
    torch.save(model.state_dict(), save_model_path)

    # save to gcs
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    model_filename = save_model_path.split("/")[-1]
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(save_model_path)

    print("ā Model saved to gcs")
    return None
