from PIL import Image
import os
import time
import numpy as np
import pandas as pd
from math import exp
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import transforms
from torchvision import datasets, transforms
import torch.optim as optim
from travel_home.params import *
from PIL import Image
from travel_home.ml_logic import utils
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.applications.resnet import ResNet152

### Git hub link https://github.com/CSAILVision/places365/
# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way
import time
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F

MINIMUM_NB_OF_IMAGES = 10

def npy_loader(path : str) -> Image:
    image = Image.fromarray(np.load(path))
    return image

def prepare_train_val_folders(data_dir : str) -> None:
    train_images_path = os.path.join(data_dir, 'train')
    val_images_path = os.path.join(data_dir, 'val')

    if (os.path.isdir(train_images_path) and os.path.isdir(val_images_path)):
        print('Train and val folders already exist')
        return None

    if not os.path.isdir(train_images_path) :
        os.mkdir(train_images_path)

    if not os.path.isdir(val_images_path):
        os.mkdir(val_images_path)

    folders_to_remove = []
    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    # loop in cellid folders
    for subdir in subdirs:
        if (subdir.split("/")[-1] == "val" or subdir.split("/")[-1] == "train"):
            continue
        files = os.scandir(subdir)
        nb_files = 0
        for path in os.scandir(subdir):
            if path.is_file():
                nb_files += 1
        if nb_files < MINIMUM_NB_OF_IMAGES:
            folders_to_remove.append(subdir)
        else:
            for index, file in enumerate(files):
                if index + 1 >= nb_files * 0.7:
                    destination_folder = os.path.join(train_images_path, file.path.split("/")[-2])
                    if not os.path.isdir(destination_folder):
                        os.mkdir(destination_folder)
                    destination = os.path.join(destination_folder, file.name)
                    os.replace(file.path, destination)
                if index + 1 < nb_files * 0.7:
                    destination_folder = os.path.join(val_images_path, file.path.split("/")[-2])
                    if not os.path.isdir(destination_folder):
                        os.mkdir(destination_folder)
                    destination = os.path.join(destination_folder, file.name)
                    os.replace(file.path, destination)

    return None

def images_transformer(x):
    data_transforms = {
        'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[x]

def prepare_input_train(data_dir : str):
    image_datasets = {x: datasets.DatasetFolder(os.path.join(data_dir, x), loader=npy_loader, extensions=['.npy'], transform=images_transformer(x)) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256, shuffle=True, num_workers=4) for x in ['train', 'val']}
    return dataloaders

features_blobs = []

def load_model():
    features_blobs.clear()

    # transfer learning : resnet18 trained on places365
    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    # checkpoint
    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # don't save the gradient of the pre-trained resnet18 model
    for param in model.parameters():
        param.requires_grad = False

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = utils.recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    # layer to predict the class -- last layer
    num_ftrs = model.fc.in_features

    class_number = len(load_class_names())

    model.fc = nn.Sequential(
          nn.Linear(num_ftrs, 800),
          nn.ReLU(inplace=True),
          nn.Linear(800, class_number),
          nn.ReLU(inplace=True)
        )

    return model

def train_model(dataloaders, model, num_epochs):
    since = time.time()

    dataset_sizes = {x: len(load_class_names()) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model params
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()
        timestamp=time.strftime("%Y%m%d-%H%H%S")
        save_model_path=os.path.join(WORKING_DIR,f"{timestamp}_{epoch}.pth")
        torch.save(model.state_dict(),save_model_path)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model

def logit2prob(logit):
  odds = exp(logit)
  prob = odds / (1 + odds)
  return(prob)

def predict(model, img_path, class_names):
    # preprocess the image
    tf = utils.returnTF()
    input_img = Image.open(img_path)
    input_img = V(tf(input_img).unsqueeze(0))
    # predict the class of the image
    outputs = model(input_img)
    _, preds = torch.max(outputs, 1)
    # prediction de la classe de l'image
    coeff = outputs.detach().numpy()[0]
    df = pd.DataFrame({'probability': coeff, 'cellid': class_names})
    df['probability'] = df['probability'].apply(logit2prob)
    df = df.sort_values(by = 'probability', ascending = False)
    df_3_most_probable = df[:3].reset_index(drop=True)
    print('👉 dataframe of the 3 most probable prediction: ')
    print(df_3_most_probable)
    print('🎉 class predicted: ', class_names[preds])
    return df_3_most_probable.to_dict()
