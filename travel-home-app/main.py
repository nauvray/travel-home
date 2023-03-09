import time
import preprocess as preproc
import utils
import pandas as pd

def preprocess(root_folder : str, csv_name : str, df : pd.DataFrame):
    # filter outdoor images
    start_time = time.time()
    df = preproc.filter_outdoor_images(df)
    print('==> Time to filter outdoor images:', round(time.time() - start_time, 1), 'seconds')

    # save images as npy files
    start_time2 = time.time()
    preproc.save_images_as_npy(root_folder, df)
    print('==> Time to save images:', round(time.time() - start_time2, 1), 'seconds')

    # remove hexa columns & save csv
    start_time3 = time.time()
    preproc.create_preproc_csv(root_folder, csv_name, df)
    print('==> Time to create csv:', round(time.time() - start_time3, 1), 'seconds')

if __name__ == '__main__':
    root_folder = '../00-data/sample/'
    csv_name = 'meta_shard_0.csv'
    # df : img lat lon hexa cellid + storage path
    df = utils.get_df_from_csv(root_folder, csv_name)
    df = utils.add_storage_path(df)

    preprocess(root_folder, csv_name, df)
