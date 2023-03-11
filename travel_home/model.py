from PIL import Image
import os
import time
import numpy as np
import pandas as pd
from math import exp
import shutil
import subprocess
from preprocess.utils import recursion_change_bn, returnTF

import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets, transforms
import torch.optim as optim
import torch.utils.data as torchdata


def npy_loader(path):
    im = Image.fromarray(np.load(path))
    return im

# data input
def creation_folder(root):

    origin = os.path.join(root, 'npy')
    train_images_path = os.path.join(root, 'train')
    val_images_path = os.path.join(root, 'val')

    source = os.path.join(origin, '.')
    cmd = 'cp -R "%s" "%s"' % (source, train_images_path)
    status = subprocess.call([cmd, source, train_images_path], shell=True)

    cmd = 'cp -R "%s" "%s"' % (source, val_images_path)
    status = subprocess.call([cmd, source, val_images_path], shell=True)


    subdirs_train = [f.path for f in os.scandir(train_images_path) if f.is_dir()]
    # print(subdirs_train)
    folders_to_remove = []
    for subdir in subdirs_train:
        # print(subdir)
        files = os.scandir(subdir)
        nb_files = 0
        for path in os.scandir(subdir):
            if path.is_file():
                nb_files += 1
        if nb_files < 3:
            folders_to_remove.append(subdir)
        else:
            for index, file in enumerate(files):
                if index+1 >= nb_files * 0.7:
                    print(file)
                    os.remove(file)

    for folder in folders_to_remove:
        shutil.rmtree(folder)

    folders_val_to_remove = []
    subdirs_val = [f.path for f in os.scandir(val_images_path) if f.is_dir()]
    folders_val_to_remove = []
    for subdir in subdirs_val:
        files = os.scandir(subdir)
        nb_files = 0
        for path in os.scandir(subdir):
            if path.is_file():
                nb_files += 1
        if nb_files < 3:
            folders_val_to_remove.append(subdir)
        else:
            for index, file in enumerate(files):
                if index+1 < nb_files * 0.7:
                    os.remove(file)
    for folder in folders_val_to_remove:
        shutil.rmtree(folder)

    return None

def input_train(data_dir):
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

    image_datasets = {x: datasets.DatasetFolder(os.path.join(data_dir, x), loader=npy_loader, extensions=['.npy'], transform=data_transforms[x])  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    return data_dir, image_datasets, dataloaders, dataset_sizes, device, class_names


def load_model2(device):
    features_blobs = []
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
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    # layer to predict the class -- last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2, False)
    #model.fc = nn.ReLU(inplace=False) #### RELU A reintégrer sans que ça bug

    model = model.to(device)

    # model params
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    return model, criterion, optimizer


def train_model2(dataloaders, dataset_sizes, device, model, criterion, optimizer, num_epochs=25):
    since = time.time()

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

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model

def logit2prob(logit):
  odds = exp(logit)
  prob = odds / (1 + odds)
  return(prob)


def predict(img, class_names):
    # preprocess the image
    tf = returnTF()
    input_img = Image.open(img)
    input_img = V(tf(input_img).unsqueeze(0))
    # predict the class of the image
    outputs = model2(input_img)
    _, preds = torch.max(outputs, 1)
    # prediction de la classe de l'image
    coeff = outputs.detach().numpy()[0]
    df = pd.DataFrame({'probability': coeff, 'sell_id': class_names})
    df['probability'] = df['probability'].apply(logit2prob)
    df = df.sort_values(by = 'probability', ascending = False)
    df = df.reset_index(drop=True)
    print('👉 dataframe of the prediction: ')
    print(df)
    print('🎉 class predicted: ', class_names[preds])
    return df

if __name__ == '__main__':
    subprocess.run('gsutil -m cp -r gs://$BUCKET_NAME/npy/ ~/code/hortense-jallot/travel-home/00-data/download', shell=True)
    data_dir = '../00-data/download/'
    creation_folder(data_dir)
    img = '../00-data/hymenoptera_data/val/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg'
    data_dir, image_datasets, dataloaders, dataset_sizes, device, class_names = input_train(data_dir)
    model2, criterion, optimizer = load_model2(device) # load model
    train_model2(dataloaders, dataset_sizes, device, model2, criterion, optimizer, num_epochs=5) # train model
    predict(img, class_names) # predict
