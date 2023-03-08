import pandas as pd
import s2cell
import s2sphere
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pathlib import Path

def reduce_sample_csv(max:int,path:str) ->None:
    formated_path = Path(path)
    df_sample_csv=pd.read_csv(formated_path)
    df_sample_csv=df_sample_csv[0:max]
    df_sample_csv.to_csv(formated_path,index=False)
    return df_sample_csv

def load_sample_csv(path:str) ->pd.DataFrame:
    path = Path(path)
    df_sample_csv=pd.read_csv(path)
    df_sample_csv['cellid']='_'
    df_sample_csv['count']=1
    df_sample_csv['zoom']=1
    return df_sample_csv

def check_output_hashed(df:pd.DataFrame) ->None:
    print((df.cellid=='_').sum())
    return

def geohashing_zoom_s2(df_sample:pd.DataFrame,start_zoom:int,end_zoom:int,threshold:int) ->pd.DataFrame:
    df_sample_csv=df_sample.copy()
    # Initialize the df to find geohash at start and start-1 zoom
    df_sample_csv[f'geohash_{start_zoom-1}'] = df_sample_csv.apply(lambda x: s2cell.lat_lon_to_cell_id(x.lat,x.lon,start_zoom-1),axis=1)
    df_sample_csv[f'geohash_{start_zoom}'] = df_sample_csv.apply(lambda x: s2cell.lat_lon_to_cell_id(x.lat,x.lon,start_zoom),axis=1)
    completed_list=[]
    # Start looping and zooming
    for zoom in range(start_zoom-1,end_zoom):
        if (df_sample_csv.cellid=='_').sum()!=0:
            if zoom > start_zoom-1:
                print((df_sample_csv.cellid=='_').sum())
                zoom_n1 = df_sample_csv[[f'geohash_{zoom-1}','count']]
                zoom_n1=zoom_n1.groupby(by=f'geohash_{zoom-1}').count().reset_index()
                zoom_n2 = df_sample_csv[[f'geohash_{zoom}','count']]
                zoom_n2=zoom_n2.groupby(by=f'geohash_{zoom}').count().reset_index()
                for i in range(len(df_sample_csv)):
                    if (df_sample_csv[f'geohash_{zoom}'][i] in list(zoom_n2[f'geohash_{zoom}'][zoom_n2['count']<threshold])
                    and i not in completed_list):
                        for j in range(len(df_sample_csv)):
                            if df_sample_csv[f'geohash_{zoom}'][j]==df_sample_csv[f'geohash_{zoom}'][i]:
                                if df_sample_csv.cellid[j]=='_':
                                    df_sample_csv.loc[j,'cellid']=df_sample_csv.loc[i,f'geohash_{zoom}']
                                    df_sample_csv.loc[j,'zoom']=zoom
                                    completed_list.append(j)
                df_sample_csv.drop(columns=[f'geohash_{zoom-1}'],inplace=True)
                df_sample_csv[f'geohash_{zoom+1}'] = df_sample_csv.apply(lambda x: s2cell.lat_lon_to_cell_id(x.lat,x.lon,zoom+1),axis=1)
            else:
                next
    for k in range(len(df_sample_csv)):
        if df_sample_csv.cellid[k]=='_':
            df_sample_csv.cellid[k]=df_sample_csv[f'geohash_{zoom+1}'][k]
    df_sample_csv.drop(columns=[f'geohash_{zoom}',f'geohash_{zoom+1}'],inplace=True)
    return df_sample_csv

def create_df_squares(df_sample_csv:pd.DataFrame) ->pd.DataFrame:
    # Create cellid with only the list of Cellid present in the df
    df_cellid=df_sample_csv[['cellid','count']].copy()
    df_cellid=df_cellid.groupby('cellid').sum()
    df_cellid.reset_index(inplace=True)
    # Create the columns
    df_cellid['top_left_lat'] = '_'
    df_cellid['top_left_lon'] = '_'
    df_cellid['top_right_lat'] = '_'
    df_cellid['top_right_lon'] = '_'
    df_cellid['bot_left_lat'] = '_'
    df_cellid['bot_left_lon'] = '_'
    df_cellid['bot_right_lat'] = '_'
    df_cellid['bot_right_lon']= '_'
    # Access to the Lat and Lng of each square for plot
    for i in range(len(df_cellid)):
        df_cellid.loc[i, 'top_left_lat'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(0)))[8:]
        df_cellid.loc[i,'top_left_lon'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(0)))[8:]
        df_cellid.loc[i,'top_right_lat'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(1)))[8:]
        df_cellid.loc[i,'top_right_lon'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(1)))[8:]
        df_cellid.loc[i,'bot_left_lat'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(2)))[8:]
        df_cellid.loc[i,'bot_left_lon'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(2)))[8:]
        df_cellid.loc[i,'bot_right_lat'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(3)))[8:]
        df_cellid.loc[i,'bot_right_lon'] = str(LatLng.from_point(Cell(CellId(int(df_cellid.cellid[i]))).get_vertex(3)))[8:]
    # Convert the lat and lng in a list and then extract only lat or only lng
    for i in range(len(df_cellid)):
        df_cellid.loc[i,'top_left_lat'] = [float(j) for j in df_cellid['top_left_lat'][i].split(',')][0]
        df_cellid.loc[i,'top_left_lon'] = [float(j) for j in df_cellid['top_left_lon'][i].split(',')][1]
        df_cellid.loc[i,'top_right_lat'] = [float(j) for j in df_cellid['top_right_lat'][i].split(',')][0]
        df_cellid.loc[i,'top_right_lon'] = [float(j) for j in df_cellid['top_right_lon'][i].split(',')][1]
        df_cellid.loc[i,'bot_left_lat'] = [float(j) for j in df_cellid['bot_left_lat'][i].split(',')][0]
        df_cellid.loc[i,'bot_left_lon'] = [float(j) for j in df_cellid['bot_left_lon'][i].split(',')][1]
        df_cellid.loc[i,'bot_right_lat'] = [float(j) for j in df_cellid['bot_right_lat'][i].split(',')][0]
        df_cellid.loc[i,'bot_right_lon'] = [float(j) for j in df_cellid['bot_right_lon'][i].split(',')][1]
    return df_cellid

def plot_squares(df_cellid):
    fig = plt.Figure()
    map = Basemap(projection='cyl', resolution = 'i', llcrnrlon=-5, \
                llcrnrlat=42, urcrnrlon=10, urcrnrlat=52)
    map.drawcoastlines()
    map.drawcountries()
    map.bluemarble()

    for i in range(len(df_cellid)):
        x_big = [df_cellid['top_left_lon'][i],df_cellid['top_right_lon'][i],df_cellid['bot_left_lon'][i],df_cellid['bot_right_lon'][i],df_cellid['top_left_lon'][i]]  # lon
        y_big = [df_cellid['top_left_lat'][i],df_cellid['top_right_lat'][i],df_cellid['bot_left_lat'][i],df_cellid['bot_right_lat'][i],df_cellid['top_left_lat'][i]]   # lat
        map.plot(x_big, y_big, color='yellow', lw=1)

    plt.show()

    return