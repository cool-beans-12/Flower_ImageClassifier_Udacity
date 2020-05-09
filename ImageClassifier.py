#Created by Sridhar Nandigam 3/13/20

#This file contains all of the code from the Jupyter notebook so that things don't get to hectic

#Imports
# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

import time
from collections import OrderedDict

from PIL import Image

import json

import numpy as np

#-------------------------------------------------------------------------------------------------------------------------------


#Dictionary to hold input values for each time of network
model_dic = {"vgg16": 25088,
             "densenet121":1024
            }

#Function to initialize neural network
#Returns: model, criterion, optimizer
def initialize_nn(inputModel = "vgg16", h1 = 6272, learningrate = 0.001):
    #parse input
    if inputModel == "vgg16":
        model = models.vgg16(pretrained = True)
    elif inputModel == "densenet121":
        model = models.densenet121(pretrained = True)
    else:
        print("Invalid input. Must be vgg16 or densenet121. Please try again")
        
    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #initialize classifier based on given parameters
    inputSize = model_dic[inputModel]
    model.classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(inputSize, h1)),
                                           ('activ1', nn.ReLU()),
                                           ('dropout1', nn.Dropout(0.3)),
                                           ('h1', nn.Linear(h1, 1568)),
                                           ('activ2', nn.ReLU()),
                                           ('dropout2', nn.Dropout(0.3)),
                                           ('h2', nn.Linear(1568, 392)),
                                           ('activ3', nn.ReLU()),
                                           ('dropout3', nn.Dropout(0.3)),
                                           ('h3', nn.Linear(392, 102)),
                                           ('output', nn.LogSoftmax(dim=1))]))

    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = learningrate)
    return model, criterion, optimizer

#-------------------------------------------------------------------------------------------------------------------------------

def train_nn(model, criterion, optimizer, train_loader, valid_loader, epochs = 6, print_every = 10, device = 'gpu'):
    if torch.cuda.is_available() and device == 'gpu':
        device = 'cuda'
        model.to('cuda')
    else:
        model.to('cpu')
    
    #Keeps track of running loss and steps
    runningLoss = 0
    steps = 0
    #start training
    for epoch in range(epochs):
        for images, labels in train_loader:
            #Move to device
            images, labels = images.to(device), labels.to(device)

            logps = model(images) #Forward pass
            loss = criterion(logps, labels) #Determine loss
            
            optimizer.zero_grad() #reset gradients to avoid buildup
            
            loss.backward() #calculate gradients
            optimizer.step() #update weights and biases
            
            runningLoss += loss.item() #calculate running loss
            steps += 1
            

            if (steps % print_every == 0):
                #Test on validation set
                testLoss = 0
                accuracy = 0
            
                model.eval() #set to evaluation mode
            
                with torch.no_grad():
                    for vimages, vlabels in valid_loader:
                        vimages, vlabels = vimages.to(device), vlabels.to(device)
                        logps = model(vimages)
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                    
                        equals = top_class == vlabels.view(*top_class.shape)
                    
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                        loss = criterion(logps, vlabels)
                        testLoss += loss.item()
            
                #Print results
                print("Epoch: {}/{}".format((epoch+1),epochs),
                      "Training Loss: {:.3f}".format(runningLoss/print_every),
                      "Testing Loss: {:.3f}".format(testLoss/len(valid_loader)),
                      "Testing Accuracy: {:.3f}".format(accuracy/len(valid_loader))
                     )
            
                runningLoss = 0
                #Set back to training mode
                model.train()
                
#-------------------------------------------------------------------------------------------------------------------------------

#Saves checkpoint
def save_checkpoint(model, train_dataset, inputModel = "vgg16", h1 = 6272, learning_rate = 0.001):
    #create checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'input_model': inputModel,
                  'input_size': model_dic[inputModel],
                  'output_size': 102,
                  'hidden_layer1': h1,
                  'learning_rate': learning_rate,
                  'state_dict':model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
    
    torch.save(checkpoint, 'checkpoint.pth')
#-------------------------------------------------------------------------------------------------------------------------------

#Loads checkpoint
def load_checkpoint(filepath = 'checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    
    model, criterion, optimizer = initialize_nn(checkpoint['input_model'], 
                                                checkpoint['hidden_layer1'], 
                                                checkpoint["learning_rate"])
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint["state_dict"])
    
    return model

#-------------------------------------------------------------------------------------------------------------------------------

def process_image(imageFile):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(imageFile)
    
    #regular transforms SHOULD work...right? Because I can't figure out PIL
    #Update: it works
    
    transformer = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    final_image = transformer(image)
    
    #returns pytorch tensor
    return final_image

#-------------------------------------------------------------------------------------------------------------------------------
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    print(type(image))
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#-------------------------------------------------------------------------------------------------------------------------------
def predict(image_path, model, k=5, device = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    inputImage = process_image(image_path)
    
    #credit to an online source for these two lines of code. I couldn't figure out why the input tensor wasn't working
    inputImage = inputImage.unsqueeze_(0)
    inputImage = inputImage.float()
    
    #convert to cuda if necessary
    if torch.cuda.is_available() and device == 'gpu':
        device = 'cuda'
        model.to('cuda')
        inputImage = inputImage.to('cuda')
    else:
        model.to('cpu')
    
    #Set to evaluation mode
    model.eval()
    
    with torch.no_grad():
        logps = model(inputImage)
        ps = torch.exp(logps)
        
        top_p, top_classes = ps.topk(k, dim=1)
        
    return top_p, top_classes

#-------------------------------------------------------------------------------------------------------------------------------