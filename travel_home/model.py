import matplotlib.pyplot as plt
from PIL import Image
import os
from utils import recursion_change_bn, returnTF, hook_feature

import torch.nn as nn
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import cv2
from torchvision import datasets, models, transforms
import torch.optim as optim


# data input
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../00-data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



features_blobs = []

def load_model2(pretrained):
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

    # couche pour extraire les features
    features_names = ['layer4','avgpool']
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    ####
    #### ajouter 2 couches relu 256 neurones
    ####

    # couche pour prédire la classe -- dernière couche
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2, False)

    model.eval()

    return model


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):

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
    return model


def predict(img):
    # preprocess the image
    tf = returnTF() # image transformer
    input_img = Image.open(img)
    input_img = V(tf(input_img).unsqueeze(0))
    # predict the class of the image
    outputs = model2(input_img)
    _, preds = torch.max(outputs, 1)
    # quelques print
    print('outputs: ', outputs)
    print('class_names: ', class_names)
    print('preds: ', preds)
    print('_: ', _)
    print('class preds: ', class_names[preds]) ###prediction de la classe de l'image
    return class_names[preds]


if __name__ == '__main__':
    model2 = load_model2(pretrained=False)
    optimizer_conv = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    train_model(model2, nn.CrossEntropyLoss, optimizer_conv, exp_lr_scheduler, num_epochs=40)
    img = '../00-data/hymenoptera_data/val/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg'
    predict(img)
