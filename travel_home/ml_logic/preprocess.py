import numpy as np
import pandas as pd
import utils
import model as resnet_model
import os
from PIL import Image
from google.cloud import storage
from io import BytesIO
from multiprocessing import Process
from travel_home.params import *
import subprocess
import time
from multiprocessing import Process
from torch.autograd import Variable as V

# nb of threads for // computing
PARALLEL_COMPUTING_BATCH_SIZE = 3


def filter_outdoor_images(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nStarting filtering...")
    outdoor_list = []
    cat_attrs_list = []
    scene_attrs_list = []

    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = utils.load_labels()
    # load the model
    model = utils.load_preproc_model()
    # load the transformer
    tf = utils.returnTF()  # image transformer

    for i in range(len(df)):
        img = utils.get_image_from_hexa(df.data[i])
        input_img = V(tf(img).unsqueeze(0))
        outdoor, cat_attrs, scene_attrs = utils.create_tags(
            input_img, model, labels_IO, labels_attribute, W_attribute, classes
        )
        outdoor_list.append(outdoor)
        cat_attrs_list.append(cat_attrs)
        scene_attrs_list.append(scene_attrs)

    df["outdoor"] = outdoor_list
    df["cat_attrs"] = cat_attrs_list
    df["scene_attrs"] = scene_attrs_list
    # filter df with outdoor flag
    df_outdoor = df[df.outdoor == 1]
    df_outdoor.reset_index(drop=True, inplace=True)
    # remove outdoor column
    df_outdoor = df_outdoor.drop(columns="outdoor")

    print(
        f"✅ filtered data frame ({df_outdoor.shape[0]} rows, {df_outdoor.shape[1]} columns)"
    )
    return df_outdoor


def save_images_as_npy(csv_path: str, df: pd.DataFrame) -> None:
    """given a data frame containing images, save images matching the list of indexes in the dataframe"""

    print(f"\nSaving images in gcs...")

    for i in range(0, len(df), PARALLEL_COMPUTING_BATCH_SIZE):
        # create all tasks
        if i + PARALLEL_COMPUTING_BATCH_SIZE < len(df):
            processes = [
                Process(
                    target=save_in_gcs_task,
                    args=(
                        df.data[j],
                        df.img[j],
                        df.cellid[j],
                        csv_path,
                    ),
                )
                for j in range(i, i + PARALLEL_COMPUTING_BATCH_SIZE)
            ]
        else:
            processes = [
                Process(
                    target=save_in_gcs_task,
                    args=(
                        df.data[j],
                        df.img[j],
                        df.cellid[j],
                        csv_path,
                    ),
                )
                for j in range(i, len(df))
            ]
        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()

    print(f"✅ npy images saved to gcs")
    # drop hexadecimal column as images are now saved in gcs...
    df.drop(columns="data", inplace=True)


def save_in_gcs_task(hexa, img_name, cellid, csv_path):
    image = Image.open(BytesIO(bytes.fromhex(hexa)))
    npy_img_name = img_name.strip(".jpg") + ".npy"
    image_array = np.array(image.getdata(), dtype=np.uint8).reshape(
        image.size[1], image.size[0], 3
    )
    # print(f"Shape of the image to save: ({image.size[1]}, {image.size[0]}, 3)")

    images_root = os.path.join(csv_path, "npy")
    if not os.path.isdir(images_root):
        os.mkdir(images_root)

    # save npy locally
    npy_local_file_path = os.path.join(images_root, npy_img_name)
    np.save(npy_local_file_path, image_array)

    # save in gcs
    storage_filename = f"npy/{cellid}/{npy_img_name}"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(npy_local_file_path)

    # clean
    os.remove(npy_local_file_path)


def create_preproc_csv(root_folder: str, csv_name: str, df: pd.DataFrame) -> None:
    print(f"\nCreating pre-processed csv...")
    csv_number = utils.get_csv_number(csv_name)
    output_csv_name = f"pre_proc_data_{csv_number}.csv"
    output_csv_path = os.path.join(root_folder, output_csv_name)

    header = False if os.path.isfile(output_csv_path) else True
    print(df.keys())
    df.to_csv(output_csv_path, mode="a", header=header, index=False)
    # save in gcs
    # storage_file_path = f"gs://{BUCKET_NAME}/preproc_csv/{output_csv_name}"
    # subprocess.call(['gsutil', 'cp', output_csv_path, storage_file_path])

    if header:
        print(f"✅ csv {output_csv_name} created!")
    else:
        print(f"✅ csv {output_csv_name} appended!")


def preprocess_csv_task(root_folder: str, csv_name: str) -> None:
    # df : img lat lon hexa cellid + storage path
    df = utils.get_df_from_csv(root_folder, csv_name)
    df = utils.add_storage_path(df)
    df = df[0:].reset_index(drop=True)
    # filter outdoor images
    start_time1 = time.time()
    df = filter_outdoor_images(df)
    df.to_csv(f"{root_folder}outdoor.csv")
    print(
        "==> Time to filter outdoor images:", round(time.time() - start_time1, 1), "s"
    )

    # save images as npy files
    start_time2 = time.time()
    save_images_as_npy(root_folder, df)
    print("==> Time to save images:", round(time.time() - start_time2, 1), "s")

    # remove hexa columns & save csv
    # create_preproc_csv(root_folder, csv_name, df)


def preprocess_csv(csv_names: list) -> None:
    for i in range(0, len(csv_names), PARALLEL_COMPUTING_BATCH_SIZE):
        # create all tasks
        if i + PARALLEL_COMPUTING_BATCH_SIZE < len(csv_names):
            processes = [
                Process(
                    target=preprocess_csv_task,
                    args=(
                        root_folder,
                        csv_names[j],
                    ),
                )
                for j in range(i, i + PARALLEL_COMPUTING_BATCH_SIZE)
            ]
        else:
            processes = [
                Process(
                    target=preprocess_csv_task,
                    args=(
                        root_folder,
                        csv_names[j],
                    ),
                )
                for j in range(i, len(csv_names))
            ]
        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()


if __name__ == "__main__":
    root_folder = "../../00-data/data_csv_hashed/"  ## TO CHANGE
    start_time = time.time()

    # To do to resave the pictures somewhere else
    for i in range(140, 141):
        df_preproc = pd.read_csv(f"../00-data/data_preproc/pre_proc_data_{i}.csv")
        df_data = pd.read_csv(f"../00-data/data_csv/meta_shard_{i}.csv")
        df_data.drop(columns=["lat", "lon"], inplace=True)
        df_preproc["storage_filename"] = df_preproc["storage_filename"].apply(
            lambda x: x.replace("bucket", "bckt")
        )
        df_preproc_data = pd.merge(df_preproc, df_data)
        save_images_as_npy(root_folder, df_preproc_data)
        df_preproc.to_csv(f"../../00-data/data_csv_hashed2/pre_proc_data_{i}.csv")
    # End to redo to resave pictures somewhere else

    # csv_names = [filename for filename in os.listdir(root_folder) if filename.startswith("meta_shard_")]
    # print(csv_names)
    # preprocess_csv(csv_names)
    print("=====> TOTAL TIME:", round((time.time() - start_time) / 60.0, 1), "mn")
