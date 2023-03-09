import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet import ResNet152
import time
# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import cv2
from PIL import Image
from google.cloud import storage
from params import *

def load_npy_image(npy_path : str, npy_file : str) -> np.ndarray:
    '''load image (numpy.array) from npy file'''
    npy_file_path = os.path.join(npy_path, npy_file)
    img_array = np.load(npy_file_path)
    print(f"npy image loaded with shape: ({img_array.shape[0]}, {img_array.shape[1]}, 3)")
    return img_array

def get_df_from_csv(csv_path : str, csv_name : str) -> pd.DataFrame:
    '''return data frame from csv file'''
    csv_file_path = os.path.join(csv_path, csv_name)
    df = pd.read_csv(csv_file_path)
    print(f"✅ data frame loaded ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df

def add_storage_path(df : pd.DataFrame) -> pd.DataFrame:
    df_edit = df.copy()
    df_edit['storage_filename'] = df_edit.apply(lambda x: f"gs://{BUCKET_NAME}/npy/" + str(x.cellid) + "/" + x.img.strip('.jpg') + '.npy', axis=1)
    return df_edit

def display_image_from_hexa(hexa : str) -> None:
    '''display image in hexadecimal format'''
    image = Image.open(BytesIO(bytes.fromhex(hexa)))
    width, height = image.size
    print(f"Shape of the image: {height} x {width}")
    plt.imshow(image)
    plt.show()

def display_image_from_npy(npy_path : str, npy_file : str) -> None:
    '''display npy image'''
    img_array = load_npy_image(npy_path, npy_file)
    plt.imshow(img_array)
    plt.show()

def resize_image(image_array : np.ndarray, target_size : tuple) -> np.ndarray:
    '''from image and target size, return resized image'''
    img = Image.fromarray(image_array)
    resized_image = img.resize(size=(target_size[0], target_size[1]))
    return np.array(resized_image)

def get_csv_number(csv_name : str) -> str :
    return csv_name.split('_')[2].strip('.csv')

def predict_image_tags(image_array : np.ndarray, tags_number : int = 10) -> None:
    '''output the tags for image using ResNet152 model'''
    resized_img_array = resize_image(image_array, (224, 224))
    model = ResNet152(weights='imagenet')
    X = np.expand_dims(resized_img_array, axis=0)
    # print("X shape", X.shape)
    X_preproc = preprocess_input(X)
    # print("X_preproc shape", X_preproc.shape)
    preds = model.predict(X_preproc)
    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', decode_predictions(preds, top=tags_number)[0])

def get_image_from_hexa(hexa : str) -> Image:
    return Image.open(BytesIO(bytes.fromhex(hexa)))

def get_image_from_url(img_url) -> Image:
    # load the test image
    os.system('wget %s -q -O test.jpg' % img_url)
    img = Image.open('test.jpg')
    return img

def get_image_from_npy(npy_path : str, npy_file : str) -> Image:
    image_array = load_npy_image(npy_path, npy_file)
    img = Image.fromarray(image_array)
    return img

########################################################################
### Git hub link https://github.com/CSAILVision/places365/
########################################################################

features_blobs = []
def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def load_model(pretrained):
    features_blobs.clear()
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    # arch = 'wideresnet152'
    # model_file = '%s_places365.pth.tar' % arch

    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(pretrained=pretrained, num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()

    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    # print("features_blobs", features_blobs)
    return model

def create_tags(input_img, model, labels_IO, labels_attribute, W_attribute, classes):
    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        # print('--TYPE OF ENVIRONMENT: indoor')
        outdoor=0
    else:
        # print('--TYPE OF ENVIRONMENT: outdoor')
        outdoor=1

    # print("outdoor:", outdoor)
    # output the prediction of scene category
    cat_attrs = ', '.join(str(probs[i]) + " -> " + str(classes[idx[i]]) for i in range(0, 5))
    # print('--SCENE CATEGORIES:', cat_attrs)

    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    scene_attrs = ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)])
    # print('--SCENE ATTRIBUTES:', scene_attrs)
    return outdoor

    # generate class activation mapping
    # print('Class activation map is saved as cam.jpg')
    # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # # render the CAM and output
    # img = cv2.imread('test.jpg')
    # height, width, _ = img.shape
    # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    # result = heatmap * 0.4 + img * 0.5
    # cv2.imwrite('cam.jpg', result)

def tag_image(img : Image) -> None:
    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()
    # load the model
    model = load_model(pretrained=False)
    # load the transformer
    tf = returnTF() # image transformer
    input_img = V(tf(img).unsqueeze(0))
    outdoor = create_tags(input_img, model, labels_IO, labels_attribute, W_attribute, classes)

    return outdoor
