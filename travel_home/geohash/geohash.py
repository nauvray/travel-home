import pandas as pd
import s2cell
import s2sphere
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from mpl_toolkits.basemap import Basemap

def reduce_sample_csv(limit_max:int,path:str) ->None:
    formated_path = Path(path)
    df_sample_csv=pd.read_csv(formated_path)
    df_sample_csv=df_sample_csv[0:limit_max]
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

def geohashing_zoom_s2(start_zoom:int,end_zoom:int,threshold:int,path:str,all_files:bool,reduced:bool,limit_max:int) ->pd.DataFrame:
    nb_files=142
    if all_files == True :
        file_path = Path(f'{path}meta_shard_no_img.csv')
        if file_path.is_file():
            df_sample=pd.read_csv(f'{path}meta_shard_no_img.csv')
            max_i = int(df_sample['folder'].max())
            df_sample=df_sample[df_sample['folder']<max_i]
            print(f'Loading the folder until file {max_i}')
        else:
            max_i=0
            print('No existing file')
        for i in range(nb_files):
            if i < max_i:
                next
            else:
                if i ==0:
                    df_sample=pd.read_csv(f'{path}meta_shard_{i}.csv')
                    df_sample['folder']=i
                    df_sample.drop(columns=['data'],inplace=True)
                    df_sample.to_csv(f'{path}meta_shard_no_img.csv',index=False)
                    print(f'Loaded {i} file')
                else:
                    df_temp=pd.read_csv(f'{path}meta_shard_{i}.csv')
                    df_temp['folder']=i
                    df_temp.drop(columns=['data'],inplace=True)
                    df_sample=pd.concat([df_sample,df_temp])
                    df_sample.to_csv(f'{path}meta_shard_no_img.csv',index=False)
                    print(f'Loaded {i} file')
                df_sample['cellid']='_'
                df_sample['count']=1
                df_sample['zoom']=1
                df_sample.to_csv(f'{path}meta_shard_no_img.csv',index=False)

    else:
        if reduced == True:
            df_sample=reduce_sample_csv(limit_max,path)
        else:
            df_sample=load_sample_csv(path)
    df_sample_csv=df_sample.copy()
    df_sample_csv.reset_index(inplace=True,drop=True)
    # Initialize the df to find geohash at start and start-1 zoom
    df_sample_csv[f'geohash_{start_zoom}'] = df_sample_csv.apply(lambda x: s2cell.lat_lon_to_cell_id(x.lat,x.lon,start_zoom),axis=1)
    completed_list=[]
    # Start looping and zooming
    for zoom in range(start_zoom-1,end_zoom):
        if (df_sample_csv.cellid=='_').sum()!=0:
            if zoom > start_zoom-1:
                print((df_sample_csv.cellid=='_').sum())
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
                df_sample_csv.drop(columns=[f'geohash_{zoom}'],inplace=True)
                df_sample_csv[f'geohash_{zoom+1}'] = df_sample_csv.apply(lambda x: s2cell.lat_lon_to_cell_id(x.lat,x.lon,zoom+1),axis=1)
            else:
                next
    for k in range(len(df_sample_csv)):
        if df_sample_csv.cellid[k]=='_':
            df_sample_csv.loc[k,'cellid']=df_sample_csv.loc[k,f'geohash_{zoom+1}']
            df_sample_csv.loc[k,'zoom']=zoom+1
    # df_sample_csv.drop(columns=[f'geohash_{zoom+1}'],inplace=True)
    for i in range(nb_files):
        df_temp = pd.read_csv(f'{path}meta_shard_{i}.csv')
        print(f'Loading file {i}')
        df_extract=df_sample_csv[df_sample_csv['folder']==i].reset_index(drop=True,inplace=False)
        df_extract=pd.concat([df_temp,df_extract['cellid']],axis=1)
        df_extract.to_csv(f'{path}../data_csv_hashed/meta_shard_{i}.csv',index=False)
        print(f'File {i} loaded in {path}../data_csv_hashed/meta_shard_{i}.csv')
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

def plot_squares(df_cellid:pd.DataFrame,path:str):
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
    plt.savefig(f'{path}Map.png')
    return
