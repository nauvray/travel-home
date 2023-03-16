import pandas as pd
import s2cell
import numpy as np



# Fonction permettant de récupérer le carré d'appartenance d'une photot (définie par lat, lon)
# du geohash de Nicolas à partir de son DataFrame.


def photo_cell_belong(df:pd.DataFrame,lat:float,lon:float):

    start_zoom = 5
    end_zoom = 16
    photo_cell_belonging=None

    for i in range(start_zoom,end_zoom + 1):
        new_photo_cell = s2cell.lat_lon_to_cell_id(lat,lon,i)
        if new_photo_cell in df['cellid'].values:
            photo_cell_belonging = new_photo_cell

    return photo_cell_belonging
