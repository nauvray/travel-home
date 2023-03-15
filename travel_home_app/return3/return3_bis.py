import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
import subprocess
from travel_home.ml_logic.utils import *
import random

########################################################################
### NOUVELLE FONCTION
########################################################################


def plot_4pics_around(cellid):
    my_local_path = '/Users/marie/code/Marie-Pierre74/travel-home/00-data/img_test'
    cellid_path =  f'gs://{BUCKET_NAME}/npy/{cellid}'
    subprocess.call(['gsutil', '-m', 'cp', '-r', cellid_path, my_local_path])

    image_path = os.path.join(my_local_path,cellid)

    nb_images = 4
    count = 1

    for i in range(nb_images):

        file_name = random.choice(os.listdir(image_path))
        img_array = load_npy_image(image_path,file_name)
        plt.subplot(nb_images,1,count)
        plt.imshow(img_array)

        #Remove plot ticks
        plt.xticks(())
        plt.yticks(())
        count +=1
    return plt.gcf()
