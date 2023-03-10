import numpy as np
import pandas as pd
import utils
import os
from PIL import Image
from google.cloud import storage
from params import *
from io import BytesIO
from multiprocessing import Process

# nb of threads for // computing
PARALLEL_COMPUTING_BATCH_SIZE = 3

def filter_outdoor_images(df : pd.DataFrame) -> pd.DataFrame:
    print(f"\nStarting filtering...")
    outdoor_list = []

    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = utils.load_labels()
    # load the model
    model = utils.load_model(pretrained=False)
    # load the transformer
    tf = utils.returnTF() # image transformer

    for i in range(len(df)):
        img = utils.get_image_from_hexa(df.data[i])
        input_img = utils.V(tf(img).unsqueeze(0))
        outdoor = utils.create_tags(input_img, model, labels_IO, labels_attribute, W_attribute, classes)
        outdoor_list.append(outdoor)

    df['outdoor'] = outdoor_list
    # filter df with outdoor flag
    df_outdoor = df[df.outdoor == 1]
    df_outdoor.reset_index(drop=True, inplace=True)
    # remove outdoor column
    df_outdoor = df_outdoor.drop(columns='outdoor')

    print(f"✅ filtered data frame ({df_outdoor.shape[0]} rows, {df_outdoor.shape[1]} columns)")
    return df_outdoor

def save_images_as_npy(csv_path : str, df : pd.DataFrame) -> None:
    '''given a data frame containing images, save images matching the list of indexes in the dataframe'''

    print(f"\nSaving images in gcs...")
    # for index in range(df.shape[0]):
    #     save_in_gcs_task(df.data[index], df.img[index], df.cellid[index], csv_path)

    for i in range(0, len(df), PARALLEL_COMPUTING_BATCH_SIZE):
        # create all tasks
        if (i + PARALLEL_COMPUTING_BATCH_SIZE < len(df)):
            processes = [Process(target=save_in_gcs_task, args=(df.data[j], df.img[j], df.cellid[j], csv_path,)) for j in range(i, i+PARALLEL_COMPUTING_BATCH_SIZE)]
        else:
            processes = [Process(target=save_in_gcs_task, args=(df.data[j], df.img[j], df.cellid[j], csv_path,)) for j in range(i, len(df))]
        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()

    print(f"✅ npy images saved to gcs")
    # drop hexadecimal column as images are now saved in gcs...
    df.drop(columns='data', inplace=True)


def save_in_gcs_task(hexa, img_name, cellid, csv_path):
    image = Image.open(BytesIO(bytes.fromhex(hexa)))
    npy_img_name = img_name.strip('.jpg') + '.npy'
    image_array = np.array(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0], 3)
    # print(f"Shape of the image to save: ({image.size[1]}, {image.size[0]}, 3)")

    image_folder = os.path.join(csv_path, "npy")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    # save npy locally
    npy_local_file_path = os.path.join(image_folder, npy_img_name)
    np.save(npy_local_file_path, image_array)

    # save in gcp
    # get cellid of the image
    cellid = str(cellid)
    storage_filename = f"npy/{cellid}/{npy_img_name}"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(npy_local_file_path)

    # clean
    os.remove(npy_local_file_path)


def create_preproc_csv(root_folder : str, csv_name : str, df : pd.DataFrame) -> None:
    print(f"\nCreating pre-processed csv...")
    csv_number = utils.get_csv_number(csv_name)
    output_csv_name = f'pre_proc_data_{csv_number}.csv'
    df.to_csv(os.path.join(root_folder, output_csv_name))
    print(f"✅ csv {output_csv_name} created!")
