import os

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_REGION = os.environ.get("REGION")
GCP_PROJECT = os.environ.get("PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
CLASS_NAMES_FILE = os.environ.get("CLASS_NAMES_FILE")
WORKING_DIR = os.environ.get("WORKING_DIR")
ENV = os.environ.get("ENV")
GCP_KEY_PATH = os.environ.get("GCP_KEY_PATH")

def load_class_names():
    data_file_path = os.path.join(os.path.dirname(__file__), CLASS_NAMES_FILE)

    with open(data_file_path, "r") as f:
        lines = f.readlines()
        cellid_list = []
        for l in lines:
            cellid_line = l.split()
            cellid_list.extend(cellid_line)
        return cellid_list
