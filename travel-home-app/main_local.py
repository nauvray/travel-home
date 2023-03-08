import preprocess as preproc
import os
import time

def preprocess(root_folder : str, csv_name : str) -> None:
    csv_number = preproc.get_csv_number(csv_name)
    df = preproc.get_df_from_csv(root_folder, csv_name)
    df = df.copy()

    # save image as npy files
    preproc.save_images_as_npy(root_folder, csv_number, df) # can give a list of indexes [19, 5, 7]

    # get npy images
    images_folder = os.path.join(root_folder, csv_number)
    npy_images = os.listdir(images_folder)

    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = preproc.load_labels()
    # load the model
    model = preproc.load_model()
    # load the transformer
    tf = preproc.returnTF() # image transformer

    for npy_image in npy_images:
        # load image and pre process it
        img = preproc.get_image_from_npy(images_folder, npy_image)
        # img = get_image_from_url('http://places.csail.mit.edu/demo/6.jpg')
        input_img = preproc.V(tf(img).unsqueeze(0))

        outdoor, cat_attrs, scene_attrs = preproc.create_tags(input_img, model, labels_IO, labels_attribute, W_attribute, classes)

        image_name = npy_image.strip('npy') + 'jpg'
        df.loc[df.img==image_name , 'Outdoor'] = outdoor
        df.loc[df.img==image_name , 'Cat_Attr'] = cat_attrs
        df.loc[df.img==image_name , 'Scene_Attr'] = scene_attrs
        print(df[df.img==image_name])

    # save df in a csv
    df.to_csv(os.path.join(images_folder, 'pre_proc_data.csv'))
    print(f"✅ csv created!")


if __name__ == '__main__':
    start_time = time.time()
    root_folder = '../00-data/sample'
    csv_name = 'meta_shard_0.csv'
    preprocess(root_folder, csv_name)

    elapsed_time = time.time() - start_time
    print('Prediction execution time:', round(elapsed_time, 1), 'seconds')
