import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.preprocessing import image
import time

def save_images_as_npy(csv_path : str, csv_name : str, image_indexes : list=None) -> None:
    '''given a csv containing images, save images matching the list of indexes in npy files (uint8 data type)'''
    df = get_df_from_csv(csv_path, csv_name)

    if image_indexes is None:
        image_indexes = list(range(2)) # with df.shape[0]

    for i in image_indexes:
        image = Image.open(BytesIO(bytes.fromhex(df.data[i])))
        ### TODO to remove after Nicolas data cleaning (problème avec les "/" dans le nom du fichier => remplacé ici par "_")
        print("Original image name: ", df.img[i])
        img_name = df.img[i].strip('.jpg').replace('/', '_')
        ###
        npy_img_name = f'{img_name}.npy'
        print("New image name: ", npy_img_name)
        npy_file_path = os.path.join(csv_path,  npy_img_name)
        image_array = np.array(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0], 3)
        print(f"Shape of the image to save: ({image.size[1]}, {image.size[0]}, 3)")
        np.save(npy_file_path, image_array)
        print(f"✅ {npy_img_name} saved!")

def load_npy_image(npy_path : str, npy_file : str) -> np.ndarray:
    '''load image (numpy.array) from npy file'''
    npy_file_path = os.path.join(npy_path, npy_file)
    img_array = np.load(npy_file_path)
    print(f"npy image loaded with shape: ({img_array.shape[0]}, {img_array.shape[1]}, 3)")
    return img_array

def get_df_from_csv(csv_path : str, file_name : str) -> pd.DataFrame:
    '''return data frame from csv file'''
    csv_file_path = os.path.join(csv_path, file_name)
    df = pd.read_csv(csv_file_path)
    print(f"✅ data frame loaded ({df.shape[0]} rows and {df.shape[1]} columns)")
    return df

def display_image_from_hexa(hexa : str) -> None:
    '''display image in hexadecimal format'''
    image = Image.open(BytesIO(bytes.fromhex(hexa)))
    width, height = image.size
    print(f"Shape of the image: {height} x {width}")
    plt.imshow(image)

def display_image_from_npy(npy_path : str, npy_file : str) -> None:
    '''display npy image'''
    img_array = load_npy_image(npy_path, npy_file)
    plt.imshow(img_array)

def resize_image(image_array : np.ndarray, target_size : tuple) -> np.ndarray:
    '''from image and target size, return resized image'''
    img = Image.fromarray(image_array)
    resized_image = img.resize(size=(target_size[0], target_size[1]))
    return np.array(resized_image)

def predict_image_tags(image_array : np.ndarray, tags_number : int = 10) -> None:
    '''output the tags for image using ResNet152 model'''
    start_time = time.time()
    resized_img_array = resize_image(image_array, (224, 224))
    model = ResNet152(weights='imagenet')
    X = np.expand_dims(resized_img_array, axis=0)
    # print("X shape", X.shape)
    X_preproc = preprocess_input(X)
    # print("X_preproc shape", X_preproc.shape)
    preds = model.predict(X_preproc)
    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', decode_predictions(preds, top=tags_number)[0])
    elapsed_time = time.time() - start_time
    print('Prediction execution time:', round(elapsed_time, 1), 'seconds')

if __name__ == '__main__':
    # save paris taj maal as npy file (index 19 in meta_shard_0.csv)
    save_images_as_npy('../00-data/sample', 'meta_shard_0.csv', [19])
    # load paris taj maal npy
    img_array = load_npy_image('../00-data/sample', 'c2_02_2145881409.npy')
    # display tags for the picture
    predict_image_tags(img_array, tags_number=5)
