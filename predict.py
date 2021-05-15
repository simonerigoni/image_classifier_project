# Uses a trained network to predict the class for an input image.
# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# - Use GPU for inference: python predict.py input checkpoint --gpu
# python predict.py
# python predict.py --image_path flowers/test/11/image_03098.jpg --checkpoint_path checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import time
from PIL import Image
import argparse


DEFAULT_DATA_DIRECTORY = 'data'
DEFAULT_TEST_IMAGE = DEFAULT_DATA_DIRECTORY + '/flowers/test/1/image_06760.jpg'
DEFAULT_CHECKPOINT_FILENAME = 'classifier/checkpoint.pth'
DEFAULT_TOP_K = 5
DEFAULT_GPU = True
FILENAME_JSON_CATEGORY = 'cat_to_name.json'
FILEPATH_JSON_CATEGORY = DEFAULT_DATA_DIRECTORY + '/' + FILENAME_JSON_CATEGORY
SIZE_CROP = 224
SIZE_RESIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def parse_input_arguments():
    '''
    Parse input arguments

    Arguments:
        None

    Returns:
        image_path (str): image path to classify. Default value DEFAULT_TEST_IMAGE
        checkpoint_path (str): checkpoint path to load the model. Default value DEFAULT_CHECKPOINT_FILENAME
        top_k (int): number of top clases to use. Default value DEFAULT_TOP_K
        category_names (str): Mapping file category label to name. Default value FILEPATH_JSON_CATEGORY
        gpu (boolean): Enable the use of GPU. Default value DEFAULT_GPU
    '''
    parser = argparse.ArgumentParser(description = "Predict using a deep neural network")
    parser.add_argument('--image_path', type = str, default = DEFAULT_TEST_IMAGE, help = 'Dataset path')
    parser.add_argument('--checkpoint_path', type = str, default = DEFAULT_CHECKPOINT_FILENAME, help = 'Path to load trained model checkpoint')
    parser.add_argument('--top_k', type = int, default = DEFAULT_TOP_K, help = 'Top K most likely classes')
    parser.add_argument('--category_names', type = str, default = FILEPATH_JSON_CATEGORY, help = 'File .json for the mapping of categories to real names')
    parser.add_argument('--gpu', action = "store_true", default = DEFAULT_GPU, help = 'Use GPU if available')

    args = parser.parse_args()
    #print(args)

    return args.image_path, args.checkpoint_path, args.top_k, args.category_names, args.gpu


def load_model_checkpoint(file_path):
    '''
    Load the model checkpoint

    Arguments:
        file_path (str): checkpoint file path

    Returns:
        model (object): model loaded from checkpoint
    '''
    checkpoint = torch.load(file_path)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['network'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(pil_image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model

    Arguments:
        pil_image (PIL.Image): PIL image 

    Returns:
        np_image (numpy.array): image in numpy array
    '''
    img_loader = transforms.Compose([transforms.Resize(SIZE_RESIZE),
                                    transforms.CenterCrop(SIZE_CROP), 
                                    transforms.ToTensor()])
    
    #pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image


def get_prediction(image_path, model, top_k_probabilities = DEFAULT_TOP_K):
    '''
    Predict the class (or classes) of an image using a trained deep learning model

    Arguments:
        image_path (str): path of the image to classify
        model (object): model to make predictions
        top_k_probabilities (int): number of top clases to use
    
    Returns:
        top_probabilities (list): top k probabilities
        top_mapped_classes (list): top k label classes
    '''
    # Use GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)

    model.to(device)
    model.eval()

    pil_image = Image.open(image_path)
    #plt.imshow(pil_image) 
    
    np_image = process_image(pil_image)
    tensor_image = torch.from_numpy(np_image)
    
    inputs = Variable(tensor_image)
    
    if torch.cuda.is_available():
        inputs = Variable(tensor_image.float().cuda())           
        
    inputs = inputs.unsqueeze(dim = 0)
    log_probabilities = model.forward(inputs)
    probabilities = torch.exp(log_probabilities)    

    top_probabilities, top_classes = probabilities.topk(top_k_probabilities, dim = 1)
    #print(top_probabilities)
    #print(top_classes)
    
    class_to_idx_inverted = {model.class_to_idx[c]: c for c in model.class_to_idx}
    top_mapped_classes = list()
    
    for label in top_classes.cpu().detach().numpy()[0]:
        top_mapped_classes.append(class_to_idx_inverted[label])
    
    return top_probabilities.cpu().detach().numpy()[0], top_mapped_classes


def get_mapping_label_name_categories(category_names):
    '''
    Load json mapping file from category label to category name

    Arguments:
        category_names (str): Mapping file category label to name

    Returns:
       category_label_to_name (dict): dictionary for mapping category label to name
    '''
    print('\t' + category_names)
    with open(category_names, 'r') as f:
        category_label_to_name = json.load(f)
        # print(category_label_to_name)
    return category_label_to_name


def load_model(category_names, checkpoint_path):
    '''
    Load model from checkpoint file

    Arguments:
        category_names (str): Mapping file category label to name
        checkpoint_path (str): checkpoint path to load the model

    Returns:
        model (object): model to make predictions
        category_label_to_name (dict): dictionary for mapping category label to name
    '''
    print('Load the model checkpoint from {}'.format(checkpoint_path))
    model = load_model_checkpoint(checkpoint_path)
    # print(model)

    print('Load category name and label mapping')
    category_label_to_name = get_mapping_label_name_categories(category_names)

    return model, category_label_to_name


def predict(image_path, checkpoint_path, top_k, category_names, gpu):
    '''
    Load the model and redict the class (or classes) of an image

    Arguments:
        image_path (str): image path to classify
        checkpoint_path (str): checkpoint path to load the model
        top_k (int): number of top clases to use
        category_names (str): Mapping file category label to name
        gpu (boolean): Enable the use of GPU

    Returns:
       None
    '''
    model, category_label_to_name = load_model(category_names, checkpoint_path)

    top_probabilities, top_classes = get_prediction(image_path, model, top_k_probabilities = top_k)

    print('Probabilities: ', top_probabilities)
    #print(top_classes)
    print('Categories:    ', [category_label_to_name[c] for c in top_classes])

    if image_path == DEFAULT_TEST_IMAGE:
        # we know the directory structure so the penultimate component of the path is the categoryh path/category/image.jpg
        path_parts = image_path.split('/')
        real_category = path_parts[-2] 
        print('True category: ', category_label_to_name[real_category])


if __name__ == "__main__":
    image_path, checkpoint_path, top_k, category_names, gpu = parse_input_arguments()
    # print(image_path)
    # print(checkpoint_path)
    # print(top_k)
    # print(category_names)
    # print(gpu)
    
    predict(image_path, checkpoint_path, top_k, category_names, gpu)
