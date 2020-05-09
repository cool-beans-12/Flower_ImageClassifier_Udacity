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

#Parse in arguments
parser = argparse.ArgumentParser(description = "Take in arguments for predict.py")

#Has to read in input file, pre-existing checkpoint, topk, category labels(cat_to_name.json)
parser.add_argument('inputFile', type = str)
parser.add_argument('--checkpoint', type = str, default = "checkpoint.pth")
parser.add_argument('--topk', type = int, default = 5)
parser.add_argument("--cat_names", type = str, default = "cat_to_name.json")
parser.add_argument("--device", type = str, default = "cpu")

userInputs = parser.parse_args()

inputFile = userInputs.inputFile
checkpoint = userInputs.checkpoint
topk = userInputs.topk
json_file = userInputs.cat_names
device = userInputs.device

#Loads in json file
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

#Load in checkpoint
model = IC.load_checkpoint(checkpoint)

#Gets highest probabilities and associated classes
probabilities, classes = IC.predict(inputFile, model, topk, device)

output_prob = np.array(probabilities[0])
output_classes = [cat_to_name[str(int(index)+1)] for index in np.array(classes[0])]

#Outputs probabilities and classes
for i in range(topk):
    print("Probability that the image is a {} is {:.3f}".format(output_classes[i], output_prob[i]))