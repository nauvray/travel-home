from folium import folium, CircleMarker
import s2cell
from geopy.geocoders import Nominatim
import requests
import os
import subprocess
from PIL import Image
import numpy as np

# FUNCTION WHICH GENERATES PHOTOS FROM A WORD
def launch_plexel(word:str):
    photo = 'https://api.unsplash.com/search/photos'
    my_params = {'query':word,'client_id':'7ANOoawIlsbj-XMwK6am_kjkYwN_w-TnsNfgz0aKHFU'}
    x = requests.get(photo,params=my_params)
    x.json()
    list_link=[]
    for i in range(len(x.json()['results'])):
        if i < 5 and x.json()['results'][i]['urls']['small'] not in list_link:
            list_link.append(x.json()['results'][i]['urls']['small'])
    return list_link


# FUNCTION TO GET THE MAP
def get_map(df_test):
    # create new dataframe with center and % of weight
    df_test[['lat','lon']] = df_test.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')
    # create a new column = weight in %
    df_test['new_proba'] = df_test['probability'].apply(lambda x: round(x*100))

    threshold = 10
    df_select = df_test[df_test.new_proba > threshold]
    geolocator = Nominatim(user_agent="my-app")
    location = geolocator.geocode("France")
    map_fr = folium.Map(location=[location.latitude, location.longitude], zoom_start=5.3)

    for lat, lon, weight in zip(df_select['lat'], df_select['lon'], df_select['new_proba']):
        CircleMarker(location=[lat, lon],
                    radius=weight/2,
                    color='blue',
                    fill=True,
                    fill_color='blue').add_to(map_fr)

    return map_fr

# FUNCTION TO DISPLAY RANDOM IMAGES OF THE GEOHASH PREDICTED
def random_images(geohash):
    subprocess.call(['gsutil', '-m', 'cp', '-r', f'gs://travel-home-bucket/npy/{geohash}/', 'im_cell_id/'])
    for i in range(2):
        npy_file = (os.listdir(f"im_cell_id/{geohash}"))[i]
        file_path = f"im_cell_id/{geohash}"
        image =  get_image_from_npy(file_path, npy_file)
        image.save(f'proposal_{i}.jpg')
    return None

# IMAGE MANIPULATION
def load_npy_image(npy_path : str, npy_file : str) -> np.ndarray:
    '''load image (numpy.array) from npy file'''
    npy_file_path = os.path.join(npy_path, npy_file)
    img_array = np.load(npy_file_path)
    print(f"npy image loaded with shape: ({img_array.shape[0]}, {img_array.shape[1]}, 3)")
    return img_array

def get_image_from_npy(npy_path : str, npy_file : str) -> Image:
    image_array = load_npy_image(npy_path, npy_file)
    img = Image.fromarray(image_array)
    return img


if __name__ == '__main__':
    # get npy images in gcs
    geohash = 5169846499198107648
    random_images(geohash)




# def random_images(geohash):
#     # get npy images in gcs
#     # list_fichier = [subprocess.call(['gsutil', 'ls', '-la', f'gs://travel-home-bucket/npy/{geohash}/'])]
#     # print(list_fichier)
#     for i in range(2):
#         output = subprocess.check_output(['gsutil', 'ls', '-la', f'gs://travel-home-bucket/npy/{geohash}/'])
#         output = str(output)
#         output = output.split('\\n')
#         output = output[i]
#         output = output.split('gs')[1].split('#')[0]
#         subprocess.call(['gsutil', '-m', 'cp', f'gs{output}', 'im_cell_id/'])
#         npy_file = (os.listdir(f"im_cell_id"))[i]
#         file_path = f"im_cell_id"
#         image =  get_image_from_npy(file_path, npy_file)
#         image.save(f'aproposal_{i}.jpg')
