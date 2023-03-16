import pandas as pd
import s2cell
import s2sphere
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import folium
import geopy
from geopy.geocoders import Nominatim



# fonction qui à partir d'un DataFrame (avec une col cellid et une col weight) retourne le centre

def bubble_plot(df_result):

########################################################################
### PART 1
########################################################################
    # create new dataframe with center and % of weight
    df_result[['lat','lon']] = df_result.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')

    # create a new column = weight in %
    df_result['new_weight'] = df_result['weight'].apply(lambda x: round(x*100))


########################################################################
### PART 2
########################################################################
    threshold = 40

    # create a new dataframe with only % > threshold

    df_select = df_result[df_result.new_weight > threshold]


########################################################################
### PART 3
########################################################################
    # plot the bubble around the center + weight
    #### Avec FOLIUM


    geolocator = Nominatim(user_agent="my-app")

    location = geolocator.geocode("France")

    map_fr = folium.Map(location=[location.latitude, location.longitude], zoom_start=7)



    for lat, lon, weight in zip(df_select['lat'], df_select['lon'], df_select['new_weight']):
        folium.CircleMarker(location=[lat, lon],
                  radius=weight/4,
                  color='blue',
                  fill=True,
                  fill_color='blue',
                  popup=folium.Popup(f'{weight}%', show=True, max_width=200)).add_to(map_fr)


    return map_fr




########################################################################
### NOUVELLE FONCTION
########################################################################

from geopy import distance

def plot_4pics_around(dfcellid,dfresult):
# dfcellid : DataFrame de Nicolas avec les colonnes cellid à un certain level (dans les squares)
#dfresult : DataFrame de Hortense avec une colonne cellid(predict) et une colonne weight
    latitudes = []
    longitudes = []
    rayon = 10

    dfresult[['lat_ref','lon_ref']] = dfresult.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')
    dfcellid[['lat','lon']] = dfcellid.apply(lambda x: s2cell.cell_id_to_lat_lon(x.cellid), axis=1, result_type='expand')

    for i in range(dfresult.shape[0]):
        for j in range(dfcellid.shape[0]):
            point_i = (dfresult['lat_ref'][i],dfresult['lon_ref'][i])
            point_j = (dfcellid['lat'][j],dfcellid['lon'][j])

            distance = distance.distance(point_i, point_j).km
            if distance <= rayon:
                latitudes.append(dfcellid['lat'][j])
                longitudes.append(dfcellid['lon'][j])


  # !!!!! dfcellid = dfcellid[]   ## --> Masque Booléen  avec seulement les lat et lon dans latitudes et longitudes
  ####### plus loin #####
    max_photo = 4
    if dfcellid.shape[0] > max_photo:
        dfcellid = dfcellid.sample(4)
    else:
        dfcellid = dfcellid


 # Pour chaque liste de 4 cellid/img
    for j in range(len(dfcellid.shape[0])):
        plt.subplot(1,len(dfcellid.shape[0]),j+1)
        image_hexa = Image.open(BytesIO(bytes.fromhex(dfcellid.data[j])))
        plt.imshow(image_hexa)
