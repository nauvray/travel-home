import time
import preprocess as preproc
import utils
import pandas as pd
import os
from multiprocessing import Process

# nb of threads for // computing
PARALLEL_COMPUTING_BATCH_SIZE = 3

def preprocess(root_folder : str, csv_name : str, df : pd.DataFrame):
    # filter outdoor images
    start_time1 = time.time()
    df = preproc.filter_outdoor_images(df)
    print('==> Time to filter outdoor images:', round(time.time() - start_time1, 1), 's')

    # save images as npy files
    start_time2 = time.time()
    preproc.save_images_as_npy(root_folder, df)
    print('==> Time to save images:', round(time.time() - start_time2, 1), 's')

    # remove hexa columns & save csv
    preproc.create_preproc_csv(root_folder, csv_name, df)

def preprocess_csv_task(root_folder : str, csv_name : str) -> None:
    # df : img lat lon hexa cellid + storage path
    df = utils.get_df_from_csv(root_folder, csv_name)
    df = utils.add_storage_path(df)
    df = df[0:10].reset_index(drop=True)
    preprocess(root_folder, csv_name, df)

def preprocess_csv(csv_names : list)-> None:
    for i in range(0, len(csv_names), PARALLEL_COMPUTING_BATCH_SIZE):
        # create all tasks
        if (i + PARALLEL_COMPUTING_BATCH_SIZE < len(csv_names)):
            processes = [Process(target=preprocess_csv_task, args=(root_folder, csv_names[j],)) for j in range(i, i+PARALLEL_COMPUTING_BATCH_SIZE)]
        else:
            processes = [Process(target=preprocess_csv_task, args=(root_folder, csv_names[j],)) for j in range(i, len(csv_names))]
        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()


if __name__ == '__main__':
    root_folder = '../../00-data/pre_process3/' ## TO CHANGE
    start_time = time.time()
    csv_names = [filename for filename in os.listdir(root_folder) if filename.startswith("meta_shard_")]
    print(csv_names)
    preprocess_csv(csv_names)
    print('=====> TOTAL TIME:', round((time.time() - start_time)/60., 1), 'mn')
