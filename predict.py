import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
from time import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sb

# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

parser.add_argument('image_path', type = str, default = './data/flower_data/test/1/image_06743.jpg',
                    help = 'Provide the path to a singe image (required)')
parser.add_argument('save_path', type = str, default = "./models/checkpoint.pth",
                    help = 'Provide the path to the file of the trained model (required)')

parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                    help = 'Use a mapping of categories to real names')
parser.add_argument('--top_k', type = int, default = 5,
                    help = 'Return top K most likely classes. Default value is 5')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

args_in = parser.parse_args()



if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ****************************")
else:
    device = torch.device("cpu")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Set the model to evaluation mode (optional, but recommended for inference)
    model.eval()

    return model

checkpoint_path = args_in.save_path
model = load_checkpoint(checkpoint_path)
model.to(device)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_img = np.array(preprocess(img))
    return processed_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0) 
    image = image.to(device)
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad ():
        output = model.forward(image)
    print("Model Output (Log-Probabilities):", output)
    #output_prob = torch.exp(output)    
    probs, indeces = output.topk(topk)    
    probs =  probs.cpu().numpy().tolist()[0]    
    indeces = indeces.cpu().numpy().tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]
    
    return probs, classes


image_path = args_in.image_path
top_k      = args_in.top_k

probs, classes = predict(image_path, model, topk=top_k)

if args_in.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print("Class name:")
    print(names)
### ------------------------------------------------------------

print("Class number:")
print(classes)
print("Probability (%):")
for idx, item in enumerate(probs):
    probs[idx] = round(item/100, 2)
print(probs)

plt.figure(figsize = (6,10))

ax    = plt.subplot(2,1,1)
image = process_image(image_path)
ax    = imshow(image, ax=ax, title=names[0])
ax.axis('off')

plt.subplot(2,1,2)
sb.barplot(x=probs, y=names, color=sb.color_palette()[0])

plt.show()


# command line usage: 
# python predict.py ./data/flower_data/test/1/image_06743.jpg ./model/checkpoint.pth --gpu
# python predict.py ./data/flower_data/test/1/image_06743.jpg ./model/checkpoint.pth --category_names cat_to_name.json --top_k 5
# python predict.py ./data/flower_data/test/1/image_06743.jpg ./model/checkpoint.pth --category_names cat_to_name.json --top_k 10 --gpu