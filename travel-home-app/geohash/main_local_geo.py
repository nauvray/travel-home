import time
import geohash.geohash as geo

def geohash_csv(start_zoom:int,end_zoom:int,threshold:int,path:str,all_files:bool,reduced:bool,limit_max:int,) ->None:
    df_geohashed=geo.geohashing_zoom_s2(start_zoom,end_zoom,threshold,path,all_files,reduced,limit_max,)
    df_cellid = geo.create_df_squares(df_geohashed)
    geo.plot_squares(df_cellid,path)
    return
if __name__ == '__main__':
    path = '../00-data/data_csv/'
    # path = 'gs://travel-home-bucket/data-csv/'
    start_zoom = 5
    end_zoom = 18
    threshold = 500
    reduced = False
    all_files=True
    limit_max=1500
    geohash_csv(start_zoom,end_zoom,threshold,path,all_files,reduced,limit_max)

    # img = utils.get_image_from_npy(root_folder, 'e3_8b_5898198058.npy')
    # utils.tag_image(img)
