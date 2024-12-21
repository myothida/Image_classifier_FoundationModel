import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import json
import numpy as np
import pandas as pd

from time import time

from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, default = './data/flower_data',
                    help = 'Provide the data directory, mandatory')
parser.add_argument('--save_dir', type = str, default = './models',
                    help = 'Provide the save directory')
parser.add_argument('--arch', type = str, default = 'resnet50',
                    help = 'resnet50 or vgg13')
# hyperparameters
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'Learning rate, default value 0.001')
parser.add_argument('--hidden_units', type = int, default = 512,
                    help = 'Number of hidden units. Default value is 512')
parser.add_argument('--epochs', type = int, default = 20,
                    help = 'Number of epochs')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

#setting values data loading
args_in = parser.parse_args()

if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ********************")
else:
    device = torch.device("cpu")

### ------------------------------------------------------------

print("------ loading data ----------------------")

data_dir  = args_in.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_datasets  = datasets.ImageFolder(test_dir,  transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
test_loader  = torch.utils.data.DataLoader(test_datasets,  batch_size = 64)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print("------ data loading finished -------------")
### ------------------------------------------------------------###

print("------ building the model ----------------")

layers        = args_in.hidden_units
learning_rate = args_in.learning_rate

num_classes = 102


if args_in.arch == 'resnet50':
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_features = model.fc.in_features
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_features, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(512, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
elif args_in.arch == 'vgg13':
    model = models.vgg13(pretrained = True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, layers)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(layers, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
else:
    raise ValueError('Model arch error.')

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("****** model arch: " + args_in.arch)
print("------ model building finished -----------")

### ------------------------------------------------------------

print("------ training the model ----------------")

epochs = args_in.epochs
start_time = time()
steps = 0
print_every  = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        steps +=1
        inputs, labels = inputs.to(device), labels.to(device)   

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{num_epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    torch.cuda.empty_cache()
print("Finished fine-tuning")

computation_time = time() - start_time
print(f"Computation Time: {computation_time} seconds")

### ------------------------------------------------------------

print("------ test the model --------------------")
model.eval()

correct = 0
total = 0
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        # Move inputs and labels to the same device as the model
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        prob, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_pred.extend(predicted.cpu())  
        y_true.extend(labels.cpu())   
        
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:0.3f}%')
### ------------------------------------------------------------

model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
              'input_size': num_features,  
              'output_size': num_classes,  
              'hidden_layers': [512],
              'model_state_dict': model.state_dict(),
              'classifier': model.classifier,
              'arch': args_in.arch
              #'opt_state_dict': optimizer.state_dict
             }

model_path = args_in.save_dir + 'checkpoint.pth'
torch.save(checkpoint, model_path)
print("------ model saved -----------------------")
