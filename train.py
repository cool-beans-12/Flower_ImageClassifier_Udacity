#Created by Sridhar Nandigam 3/13/20

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt

import time
from collections import OrderedDict

from PIL import Image

import json

import numpy as np

import argparse

import ImageClassifier as IC

#This is where the fun begins
parser = argparse.ArgumentParser(description = "Take in arguments for train.py")

#Has to read in architecture, learning rate, hidden units, and training epochs

#Choose from either densenet121 or vgg16
parser.add_argument('data_dir', type = str, default = "flowers")
parser.add_argument('architecture', type = str, default = "vgg16")
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--hidden_units', type = int, default = 6272)
parser.add_argument("--epochs", type = int, default = 6)
parser.add_argument("--device", type = str, default = "gpu")
parser.add_argument("--checkpoint", type = str, default = "checkpoint.pth")
parser.add_argument("--print_every", type = int, default = 10)
userInputs = parser.parse_args()

data_dir = userInputs.data_dir
arch = userInputs.architecture
lr = userInputs.learning_rate
h1 = userInputs.hidden_units
epochs = userInputs.epochs
device = userInputs.device
checkpoint = userInputs.checkpoint
print_every = userInputs.print_every


#Loads in data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)

#Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32, shuffle = True)

#initialize neural network
model, criterion, optimizer = IC.initialize_nn(arch, h1, lr)

#train neural network
IC.train_nn(model, criterion, optimizer, train_loader, valid_loader, epochs, print_every, device)

#save model as checkpoint
IC.save_checkpoint(model, train_dataset, arch, h1, lr)

print("It's over. It's finally over")
