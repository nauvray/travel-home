from folium import folium, CircleMarker
import s2cell
from geopy.geocoders import Nominatim
from io import BytesIO
import os
import subprocess
from travel_home.ml_logic.utils import get_image_from_npy


# FUNCTION TO GET THE MAP
def get_map(df_test):
    # create new dataframe with center and % of weight
    df_test[['lat','lon']] = df_test.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')
    # create a new column = weight in %
    df_test['new_weight'] = df_test['weight'].apply(lambda x: round(x*100))

    threshold = 10
    df_select = df_test[df_test.new_weight > threshold]
    geolocator = Nominatim(user_agent="my-app")
    location = geolocator.geocode("France")
    map_fr = folium.Map(location=[location.latitude, location.longitude], zoom_start=5.3)

    for lat, lon, weight in zip(df_select['lat'], df_select['lon'], df_select['new_weight']):
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
