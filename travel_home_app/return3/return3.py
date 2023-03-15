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
    threshold = 10

    # create a new dataframe with only % > threshold

    df_select = df_result[df_result.new_weight > threshold]


########################################################################
### PART 3
########################################################################
    # plot the bubble around the center + weight
    #### Avec FOLIUM


    geolocator = Nominatim(user_agent="my-app")

    location = geolocator.geocode("France")

    map_fr = folium.Map(location=[location.latitude, location.longitude], zoom_start=6)



    for lat, lon, weight in zip(df_select['lat'], df_select['lon'], df_select['new_weight']):
        folium.CircleMarker(location=[lat, lon],
                  radius=weight/2,
                  color='blue',
                  fill=True,
                  fill_color='blue').add_to(map_fr)


    return map_fr
