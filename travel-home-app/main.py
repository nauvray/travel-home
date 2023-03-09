import time
import preprocess as preproc
import geohash.geohash as geo

def preprocess(root_folder : str, csv_name : str):
    # filter outdoor images
    start_time = time.time()
    df = preproc.filter_outdoor_images(root_folder, csv_name)
    print('==> Time to filter outdoor images:', round(time.time() - start_time, 1), 'seconds')

    # save images as npy files
    start_time2 = time.time()
    preproc.save_images_as_npy(root_folder, df)
    print('==> Time to save images:', round(time.time() - start_time2, 1), 'seconds')

    # remove hexa columns & save csv
    start_time3 = time.time()
    preproc.create_preproc_csv(root_folder, csv_name, df)
    print('==> Time to create csv:', round(time.time() - start_time3, 1), 'seconds')

def geohash_csv(start_zoom:int,end_zoom:int,threshold:int,path:str,reduced:bool,limit_max:int) ->None:
    df_geohashed=geo.geohashing_zoom_s2(start_zoom,end_zoom,threshold,path,reduced,limit_max)
    df_cellid = geo.create_df_squares(df_geohashed)
    geo.plot_squares(df_cellid)
    return

if __name__ == '__main__':
    root_folder = '../00-data/sample/'
    csv_name = 'meta_shard_0.csv'

    preprocess(root_folder, csv_name)

    # img = utils.get_image_from_npy(root_folder, 'e3_8b_5898198058.npy')
    # utils.tag_image(img)
