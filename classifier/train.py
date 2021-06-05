# Train a new network on a dataset and save the model as a checkpoint. 
# During training prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# - Choose architecture: python train.py data_dir --arch "vgg13"
# - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# - Use GPU for training: python train.py data_dir --gpu
# python train.py
# python train.py --data_directory flowers --save_directory checkpoints --network "vgg16" --learning_rate 0.001 --hidden_units 1024 --epochs 10 --gpu

import os
import numpy as np
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
import argparse


DEFAULT_DATA_DIRECTORY = '../data'
DEFAULT_TRAIN_DIRECTORY = 'flowers/train'
DEFAULT_TEST_DIRECTORY = 'flowers/valid'
DEFAULT_VALID_DIRECTORY = 'flowers/test'
DEFAULT_MODEL_DIRECTORY = None
DEFAULT_CHECKPOINT_FILENAME = 'checkpoint.pth'
DEFAULT_NETWORK = 'vgg16'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_UNITS = 1024
DEFAULT_EPOCHS = 10
DEFAULT_GPU = True
FILENAME_JSON_CATEGORY = 'cat_to_name.json'
DEGREES_ROTATION = 30
SIZE_CROP = 224
SIZE_RESIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
DROPOUT_PROBABILITY = 0.5
IN_FEATURES = 25088


def parse_input_arguments():
    '''
    Parse input arguments

    Arguments:
        None

    Returns:
        data_directory (str): directory were data is stored. Default value DEFAULT_DATA_DIRECTORY
        save_directory (str): directory to save checkpoint. Default value DEFAULT_MODEL_DIRECTORY
        network (str): network architecture to use. Default value DEFAULT_NETWORK
        learning_rate (float): learning rate to use. Default value DEFAULT_LEARNING_RATE
        hidden_units (int): hidden units to use. Default value DEFAULT_HIDDEN_UNITS
        epochs (int): epochs to use. Default value DEFAULT_EPOCHS
        gpu (boolean): Enable the use of GPU. Default value DEFAULT_GPU
    '''
    parser = argparse.ArgumentParser(description = "Train a deep neural network")
    parser.add_argument('--data_directory', type = str, default = DEFAULT_DATA_DIRECTORY, help = 'Dataset path')
    parser.add_argument('--save_directory', type = str, default = DEFAULT_MODEL_DIRECTORY, help = 'Path to save trained model checkpoint')
    parser.add_argument('--network', type = str, default = DEFAULT_NETWORK, choices = ['vgg11', 'vgg13', 'vgg16', 'vgg19'], help = 'Model architecture')
    parser.add_argument('--learning_rate', type = float, default = DEFAULT_LEARNING_RATE, help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, default = DEFAULT_HIDDEN_UNITS, help = 'Number of hidden units')
    parser.add_argument('--epochs', type = int, default = DEFAULT_EPOCHS, help = 'Number of epochs')
    parser.add_argument('--gpu', action = "store_true", default = DEFAULT_GPU, help = 'Use GPU if available')

    args = parser.parse_args()
    #print(args)
    return args.data_directory, args.save_directory, args.network, args.learning_rate, args.hidden_units, args.epochs, args.gpu


def get_data_directories(data_directory):
    '''
    Get the directories were train, validation and test data are stored

    Arguments:
        data_directory (str): base data directory

    Returns:
        train_directory (str): directory for training dataset
        valid_directory (str): directory for validation dataset
        test_directory (str): directory for test dataset
    ''' 
    train_directory = data_directory + '/' + DEFAULT_TRAIN_DIRECTORY
    valid_directory = data_directory + '/' + DEFAULT_VALID_DIRECTORY
    test_directory = data_directory + '/' + DEFAULT_TEST_DIRECTORY

    # print(data_directory)
    print('\t' + train_directory)
    print('\t' + valid_directory)
    print('\t' + test_directory)
    return train_directory, valid_directory, test_directory


def get_number_of_classes(train_directory, valid_directory, test_directory):
    '''
    Get the number of classes from the directory

    Arguments:
        train_directory (str): directory for training dataset
        valid_directory (str): directory for validation dataset
        test_directory (str): directory for test dataset

    Returns:
       number_train_classes (int): number of classes
    ''' 
    number_train_classes = len(os.listdir(train_directory))
    number_valid_classes = len(os.listdir(valid_directory))
    number_test_classes = len(os.listdir(test_directory))

    #print(number_train_classes)
    #print(number_valid_classes)
    #print(number_test_classes)

    if (number_train_classes != number_valid_classes) or (number_train_classes != number_test_classes) or (number_valid_classes != number_test_classes):
        print('Error: number of train, valid and test classes is not the same')
        exit()

    return number_train_classes


def load_datasets(train_directory, valid_directory, test_directory):
    '''
    Load datasets

    Arguments:
        train_directory (str): directory for training dataset
        valid_directory (str): directory for validation dataset
        test_directory (str): directory for test dataset

    Returns:
       train_data (torchvision.datasets.ImageFolder): train dataset
       valid_data (torchvision.datasets.ImageFolder): validation dataset
       test_data (torchvision.datasets.ImageFolder): test dataset
    '''
    train_transforms = transforms.Compose([transforms.RandomRotation(DEGREES_ROTATION),
                                        transforms.RandomResizedCrop(SIZE_CROP),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    valid_transforms = transforms.Compose([transforms.Resize(SIZE_RESIZE), 
                                        transforms.CenterCrop(SIZE_CROP),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    test_transforms = transforms.Compose([transforms.Resize(SIZE_RESIZE), 
                                        transforms.CenterCrop(SIZE_CROP),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_directory, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_directory, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_directory, transform = test_transforms)

    return train_data, valid_data, test_data


def get_data_loaders(train_data, valid_data, test_data):
    '''
    Get data loaders

    Arguments:
       train_data (torchvision.datasets.ImageFolder): train dataset
       valid_data (torchvision.datasets.ImageFolder): validation dataset
       test_data (torchvision.datasets.ImageFolder): test dataset

    Returns:
       train_loader (torch.utils.data.DataLoader): train data loader
       valid_loader (torch.utils.data.DataLoader): validation data loader
       test_loader (torch.utils.data.DataLoader): test data loader
    '''

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

    return train_loader, valid_loader, test_loader


def get_mapping_label_name_categories(data_directory):
    '''
    Load json mapping file from category label to category name

    Arguments:
        category_names (str): Mapping file category label to name

    Returns:
       category_label_to_name (dict): dictionary for mapping category label to name
    '''
    mapping_file = data_directory + '/' + FILENAME_JSON_CATEGORY
    print('\t' + mapping_file)
    with open(mapping_file, 'r') as f:
        category_label_to_name = json.load(f)
        # print(category_label_to_name)
    return category_label_to_name


def get_pretrained_model(network, number_classes, hidden_units):
    '''
    Get pretrained model and adapt it to current needs

    Arguments:
        network (str): network architecture to use
        number_classes (int): number of classes

    Returns:
       model (object): model adapted to current needs
       classifier (object): classifier adapted to current needs
    '''
    model = getattr(torchvision.models, network)(pretrained = True)
    out_features = hidden_units

    #print(model)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(IN_FEATURES, out_features)),
                                            ('drop', nn.Dropout(p = DROPOUT_PROBABILITY)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(out_features, number_classes)),
                                            ('output', nn.LogSoftmax(dim = 1))
                                            ]))
        
    model.classifier = classifier
    #print(model)
    return model, classifier

def save_model_checkpoint(model, train_data, network, number_classes, learning_rate, classifier, epochs, optimizer, save_path, checkpoint_filename):
    '''
    Save model checkpoint

    Arguments:
        network (str): network architecture used
        number_classes (int): number of classes

    Returns:
       None
    '''
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'network': network,
                'input_size': IN_FEATURES,
                'output_size': number_classes,
                'learning_rate': learning_rate,       
                'batch_size': BATCH_SIZE,
                'classifier' : classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_path)

def train(data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu):
    '''
    Train a network and save it in a checkpoint file

    Arguments:
        data_directory (str): directory were data is stored
        save_directory (str): directory to save checkpoint
        network (str): network architecture to use
        learning_rate (float): learning rate to use
        hidden_units (int): hidden units to use
        epochs (int): epochs to use
        gpu (boolean): Enable the use of GPU

    Returns:
       None
    '''
    print('Get data directories')
    train_directory, valid_directory, test_directory = get_data_directories(data_directory)

    print('Get the number of classes')
    number_classes = get_number_of_classes(train_directory, valid_directory, test_directory)

    print('Load datasets')
    train_data, valid_data, test_data = load_datasets(train_directory, valid_directory, test_directory)

    print('Get data loaders')
    train_loader, valid_loader, test_loader = get_data_loaders(train_data, valid_data, test_data)

    print('Load category name and label mapping')
    category_label_to_name = get_mapping_label_name_categories(data_directory)
    
    print('Download pretrained model')
    model, classifier = get_pretrained_model(network, number_classes, hidden_units)

    # Train the network

    if gpu == True:
        # Use GPU if it's available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    print('Using:', device)

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device)

    validation_step = True

    print('Training started')
    start_time = time.time()

    for epoch in range(epochs):
        train_loss = 0
        print('Move input and label tensors to the default device')
        for inputs, labels in train_loader:     
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_probabilities = model.forward(inputs)
            loss = criterion(log_probabilities, labels)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
        
        print('\nEpoch: {}/{} '.format(epoch + 1, epochs), '\n    Training:\n      Loss: {:.4f}  '.format(train_loss / len(train_loader)))
            
        if validation_step == True:
            
            valid_loss = 0
            valid_accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    log_probabilities = model.forward(inputs)
                    loss = criterion(log_probabilities, labels)
            
                    valid_loss = valid_loss + loss.item()
            
                    # Calculate accuracy
                    probabilities = torch.exp(log_probabilities)
                    top_probability, top_class = probabilities.topk(1, dim = 1)
                    
                    equals = top_class == labels.view(*top_class.shape)
                    
                    valid_accuracy = valid_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
            
            model.train()
        
            print('\n    Validation:\n      Loss: {:.4f}  '.format(valid_loss / len(valid_loader)), 'Accuracy: {:.4f}'.format(valid_accuracy / len(valid_loader)))
            
    end_time = time.time()
    print('Training ended')

    training_time = end_time - start_time
    print('\nTraining time: {:.0f}m {:.0f}s'.format(training_time / 60, training_time % 60))

    # Do validation on the test set
    print('Validation on the test set')
    test_loss = 0
    test_accuracy = 0
    model.eval()
    start_time = time.time()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        log_probabilities = model.forward(inputs)
        loss = criterion(log_probabilities, labels)

        test_loss = test_loss + loss.item()

        # Calculate accuracy
        probabilities = torch.exp(log_probabilities)
        top_probability, top_class = probabilities.topk(1, dim = 1)

        equals = top_class == labels.view(*top_class.shape)

        test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()

    print('\nTest:\n  Loss: {:.4f}  '.format(test_loss / len(test_loader)), 'Accuracy: {:.4f}'.format(test_accuracy / len(test_loader)))

    end_time = time.time()
    print('Validation ended')
    validation_time = end_time - start_time
    print('Validation time: {:.0f}m {:.0f}s'.format(validation_time / 60, validation_time % 60))

    # Save model checkpoint
    save_path = ''

    if save_directory == None:
        save_path = DEFAULT_CHECKPOINT_FILENAME
    else:
        save_path = save_directory + '/' + DEFAULT_CHECKPOINT_FILENAME
    
    save_model_checkpoint(model, train_data, network, number_classes, learning_rate, classifier, epochs, optimizer, save_path, DEFAULT_CHECKPOINT_FILENAME)
    print('Save the checkpoint in {}'.format(save_path))      


if __name__ == "__main__":
    data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu = parse_input_arguments()
    # print(data_directory)
    # print(save_directory)
    # print(network)
    # print(learning_rate)
    # print(hidden_units)
    # print(epochs)
    # print(gpu)
    
    train(data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu)
    