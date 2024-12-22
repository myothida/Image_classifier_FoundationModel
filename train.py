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

# Parse arguments
parser = argparse.ArgumentParser(description='Train a flower classifier.')
parser.add_argument('data_directory', type=str, help='Directory containing the dataset.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
args = parser.parse_args()

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print(f"Using device: {device}")

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = models.resnet50(weights='IMAGENET1K_V1')
num_classes = 102

num_features = model.fc.in_features

model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(num_features, args.hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(args.hidden_units, num_classes)),
    ('output', nn.LogSoftmax(dim=1))
]))

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

num_epochs = args.epochs
start_time = time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    validation_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            validation_loss += batch_loss.item()

            _, top_class = logps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{num_epochs}.. "
          f"Train loss: {running_loss/len(train_loader):.3f}.. "
          f"Validation loss: {validation_loss/len(valid_loader):.3f}.. "
          f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

    torch.cuda.empty_cache()

computation_time = time() - start_time
print(f"Computation Time: {computation_time} seconds")

model.eval()
accuracy = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        probs = model(inputs)
        top_p, top_class = probs.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Final Test accuracy: {accuracy/len(test_loader):.3f}")

model.class_to_idx = train_datasets.class_to_idx
checkpoint = {
    'class_to_idx': model.class_to_idx,
    'input_size': num_features,
    'output_size': num_classes,
    'hidden_layers': [args.hidden_units],
    'model_state_dict': model.state_dict()
}

model_path = "./models/resNet50_model_script.pth"
torch.save(checkpoint, model_path)
