import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import argparse
import json
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sb

def process_image(image_path):
    """
    Preprocesses the image for the model.
    """
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    processed_img = np.array(preprocess(img))
    return preprocess(img)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, weights_only=True)  
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    model = models.resnet50(weights='IMAGENET1K_V1')
    # Recreate the model with nn.Sequential based on the saved architecture
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_layers[0])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_layers[0], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Load the saved model state_dict (weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    return model

def predict(image_path, model, topk=5):
    model.eval()
    img = process_image(image_path).unsqueeze_(0)
    with torch.no_grad():
        output = model(img)        
        top_p, top_class = output.topk(topk, dim=1)
    return top_p.numpy()[0], top_class.numpy()[0]

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute(1, 2, 0).numpy() 
    #image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


parser = argparse.ArgumentParser(description='Predict flower name from an image.')
parser.add_argument('image_path', type=str, default = './data/flower_data/test/1/image_06743.jpg', help='Path to the input image.')
parser.add_argument('checkpoint', type=str, default = './models/resNet50_model_script.pth', help='Path to the model.')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
parser.add_argument('--category_names', type=str, default = 'cat_to_name.json', help='Path to a JSON file mapping categories to real names.')
args = parser.parse_args()

model = load_checkpoint(args.checkpoint)


if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = None

probs, classes = predict(args.image_path, model, args.top_k)
for idx, item in enumerate(probs):
    probs[idx] = round(item/100, 2)

names = [cat_to_name[str(key+1)] for key in classes]

print(probs)
print(classes)
print(names)

plt.figure(figsize = (6,10))

ax    = plt.subplot(2,1,1)
image = process_image(args.image_path)
ax    = imshow(image, ax=ax, title=names[0])
ax.axis('off')

plt.subplot(2,1,2)
sb.barplot(x=probs, y=names, color=sb.color_palette()[0])

plt.show()
