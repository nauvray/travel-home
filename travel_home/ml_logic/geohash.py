import pandas as pd
import s2cell
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.basemap import Basemap
import os

def load_sample_csv(path:str) ->pd.DataFrame:
    '''
    Loading one or several file into a DataFrame.
    If the chosen path is targetting a file, only this file will be loaded.
    If the path is targetting a folder all the csv files will be loaded only.
    The function return a DataFrame
    '''
    path = Path(path)
    if path.endswith('.csv'):
        print('Start loading one file')
        df_sample_csv=pd.read_csv(path)
        df_sample_csv['cellid']='_'
        df_sample_csv['count']=1
        df_sample_csv['zoom']=1
    else:
        for i in range(len(os.listdir(path))):
            file_list = os.listdir(path)
            file_list_csv= [filename for filename in file_list if filename.endswith('.csv') ]
            print(f'Start loading {len(file_list_csv)} files')
            if i ==0:
                df_sample_csv = pd.read_csv(file_list_csv[i])
                df_sample_csv.drop(columns=['data'],inplace=True)
                print(f'File {i} loaded')
            else:
                df_temp = pd.read_csv(file_list_csv[i])
                df_temp.drop(columns=['data'],inplace=True)
                df_sample_csv=pd.concat([df_sample_csv,df_temp],axis=0)
                print(f'File {i} loaded')
    return df_sample_csv

def check_output_hashed(df:pd.DataFrame) ->None:
    '''
    Checking that the DataFrame gave a cellid to every picture
    '''
    print((df.cellid=='_').sum())
    return

def geohashing_zoom_s2(start_zoom:int,end_zoom:int,threshold:int,path:str,all_files:bool) ->pd.DataFrame:
    '''
    Geohashing the region of the world linked to the dataset put as input. Here the dataset have been restricted to a square around France.
    Latitude are constrained between 42 and 52, and longitude between -5 and 10.
    First step is to call all the csv, load them and save them without the "data" column as an intermediate state.
    Second step is to have a look at the global map, from "start_zoom" with the corresponding cellid and count how many photo there are in each square.
    If there are more than "threshold" then the zoom increase, if not we close the cellid and give the cellid to each photo in the square.
    Zoom steps are stopped at the "end_zoom" value.
    Third step is used to save the cellid in every original csv.
    '''
    # Loading the files in DF and saving the csv without data
    file_list = os.listdir(path)
    file_list_csv= [filename for filename in file_list if filename.endswith('.csv') ]
    nb_files=len(file_list_csv)
    print(nb_files)
    if all_files == True :
        file_path = (f'{path}meta_shard_no_img.csv')
        print(Path(file_path).is_file())
        if Path(file_path).is_file():
            df_sample=pd.read_csv(f'{path}meta_shard_no_img.csv')
            max_i = int(df_sample['folder'].max())
            df_sample=df_sample[df_sample['folder']<max_i]
            print(f'Loading the folder until file {max_i}')
        else:
            max_i=0
            print('No existing file')
        for i in range(nb_files):
            print(max_i)
            if i <= max_i:
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
        df_sample=load_sample_csv(path)
    # Copy the df
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
    # Saving the files with the adapted cellid depending on the geohash
    for k in range(len(df_sample_csv)):
        if df_sample_csv.cellid[k]=='_':
            df_sample_csv.loc[k,'cellid']=df_sample_csv.loc[k,f'geohash_{zoom+1}']
            df_sample_csv.loc[k,'zoom']=zoom+1
    df_sample_csv.to_csv(f'{path}meta_shard_no_img_zoom.csv')
    for i in range(nb_files):
        df_temp = pd.read_csv(f'{path}meta_shard_{i}.csv')
        print(f'Loading file {i}')
        df_extract=df_sample_csv[df_sample_csv['folder']==i].reset_index(drop=True,inplace=False)
        df_extract=pd.concat([df_temp,df_extract['cellid']],axis=1)
        path_hashed = path+'data_csv_hashed/'
        if Path(path_hashed).is_dir :
            df_extract.to_csv(f'{path}data_csv_hashed/meta_shard_{i}.csv',index=False)
            print(f'File {i} loaded in {path}data_csv_hashed/meta_shard_{i}.csv')
        else:
            os.mkdir(path_hashed)
            df_extract.to_csv(f'{path}data_csv_hashed/meta_shard_{i}.csv',index=False)
            print(f'File {i} loaded in {path}data_csv_hashed/meta_shard_{i}.csv')
    return df_sample_csv

def create_df_squares(df_sample_csv:pd.DataFrame) ->pd.DataFrame:
    '''
    Find the coordinates of the squares of each cellid to prepare the mapping of the squares.
    '''
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
    '''
    From the previous dataframe in create_df_squares we plot each square on a map using Basemap
    '''
    map = Basemap(projection='cyl', resolution = 'i', llcrnrlon=-5, \
                llcrnrlat=42, urcrnrlon=10, urcrnrlat=52)
    map.drawcoastlines()
    map.drawcountries()
    map.bluemarble()

    for i in range(len(df_cellid)):
        x_big = [df_cellid['top_left_lon'][i],df_cellid['top_right_lon'][i],df_cellid['bot_left_lon'][i],df_cellid['bot_right_lon'][i],df_cellid['top_left_lon'][i]]  # lon
        y_big = [df_cellid['top_left_lat'][i],df_cellid['top_right_lat'][i],df_cellid['bot_left_lat'][i],df_cellid['bot_right_lat'][i],df_cellid['top_left_lat'][i]]   # lat
        map.plot(x_big, y_big, color='yellow', lw=1)
    # plt.show()
    plt.savefig(f'{path}Map.png')
    return

def geohash_csv(start_zoom:int,end_zoom:int,threshold:int,path:str,all_files:bool) ->None:
    '''
    Global function calling successively the function to :
        - Geohash an area
        - Create a df with the coordinates of the corner of each cellid
        - Plot these squares and save the image
    '''
    df_geohashed=geohashing_zoom_s2(start_zoom,end_zoom,threshold,path,all_files)
    df_cellid =create_df_squares(df_geohashed)
    plot_squares(df_cellid,path)
    return

if __name__ == '__main__':
    # path = '../../00-data/data_csv/'
    path = 'gs://travel-home-bckt/data-csv/'
    start_zoom = 6
    end_zoom = 12
    threshold = 1000
    reduced = False
    all_files=True
    limit_max=1500
    geohash_csv(start_zoom,end_zoom,threshold,path,all_files)
