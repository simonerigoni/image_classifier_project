# Image classifier application
# python image_classifier_project.py   

import os
import argparse
import classifier.train
import predict


DEFAULT_DATA_DIRECTORY = 'data'
DEFAULT_TRAIN_DIRECTORY = DEFAULT_DATA_DIRECTORY + '/flowers/train'
DEFAULT_TEST_DIRECTORY = DEFAULT_DATA_DIRECTORY + '/flowers/valid'
DEFAULT_VALID_DIRECTORY = DEFAULT_DATA_DIRECTORY + '/flowers/test'
DEFAULT_TEST_IMAGE = DEFAULT_DATA_DIRECTORY + '/flowers/test/1/image_06760.jpg'
DEFAULT_CHECKPOINT_FILENAME = 'classifier/checkpoint.pth'
DEFAULT_FILEPATH_JSON_CATEGORY = DEFAULT_DATA_DIRECTORY + '/cat_to_name.json'
DEFAULT_MODEL_DIRECTORY = 'classifier'
DEFAULT_NETWORK = 'vgg16'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_UNITS = 1024
DEFAULT_EPOCHS = 10
DEFAULT_TOP_K = 5
DEFAULT_GPU = True


def parse_input_arguments():
    '''
    Parse the command line arguments

    Arguments:
        None

    Returns:
        image (str): image path to be classified
    '''
    parser = argparse.ArgumentParser(description = "Image Classfier")
    parser.add_argument('--image', type = str, default = DEFAULT_TEST_IMAGE, help = 'Image to classify')
    args = parser.parse_args()
    #print(args)
    return args.image


def load_classifier(checkpoint_filename = DEFAULT_CHECKPOINT_FILENAME):
    '''
    Load the classifier. Train the network if needed

    Arguments:
        checkpoint_filename (str): checkpoint filename for loading the model

    Returns:
        model (object): model loaded from checkpoint
        category_label_to_name (dict): dictionary for mapping category label to name
    '''
    print('Check if checkpoint model present\n\tCheckpoint: {}'.format(checkpoint_filename))
    if os.path.isfile(checkpoint_filename) == False:
        print('Not present. Training the model')
        classifier.train.train(DEFAULT_DATA_DIRECTORY, DEFAULT_MODEL_DIRECTORY, DEFAULT_NETWORK, DEFAULT_LEARNING_RATE, DEFAULT_HIDDEN_UNITS, DEFAULT_EPOCHS, DEFAULT_GPU)
    else:
        print('Ok')

    model, category_label_to_name = predict.load_model(DEFAULT_FILEPATH_JSON_CATEGORY, checkpoint_filename)

    return model, category_label_to_name


def get_prediction(model, category_label_to_name, image_path):
    '''
    Get the prediction probability for the image given in input

    Arguments:
        model (object): model to make predictions
        category_label_to_name (dict): dictionary for mapping category label to name
        image_path: path of the image to classify

    Returns:
        probability_category (dict): dictonary with predicted category and probability
    '''
    top_probabilities, top_classes = predict.get_prediction(image_path, model, top_k_probabilities = DEFAULT_TOP_K)
    probability_category = []
    for i in range(len(top_probabilities)):
        probability_category.append((top_probabilities[i], category_label_to_name[top_classes[i]]))
    return dict((category, probability) for probability, category in probability_category) 


def get_number_of_classes():
    '''
    Return the number of classes used

    Arguments:
        None

    Returns:
        number_of_classes (int): number of classes used
    '''
    return classifier.train.get_number_of_classes(DEFAULT_TRAIN_DIRECTORY, DEFAULT_VALID_DIRECTORY, DEFAULT_TEST_DIRECTORY)


def get_sample_from_training_dataset(category_label_to_name):
    '''
    Get a sample from the training dataset

    Arguments:
        category_label_to_name (dict): dictionary for mapping category label to name

    Returns:
        category_sample (dict): dictonary with category and image path
    '''
    directories = os.listdir(DEFAULT_TRAIN_DIRECTORY)
    category_sample = []
    for directory in directories:
        files = os.listdir(DEFAULT_TRAIN_DIRECTORY + '/' + directory)
        category_sample.append((category_label_to_name[directory], DEFAULT_TRAIN_DIRECTORY + '/' + directory + '/' + files[0]))
    return dict((category, sample) for category, sample in category_sample) 


#TODO: can be intresting to plot how many images we have for each class in the training dataset. Is the dataset unbalanced?


if __name__ == '__main__':
    image_path = parse_input_arguments()
    model, category_label_to_name = load_classifier()

    print('Number of categories: {}'.format(get_number_of_classes()))
    #print(get_sample_from_training_dataset(category_label_to_name))

    category_probability = get_prediction(model, category_label_to_name, image_path)
    print(category_probability)

    if image_path == DEFAULT_TEST_IMAGE:
        # we know the directory structure so the penultimate component of the path is the categoryh path/category/image.jpg
        path_parts = image_path.split('/')
        real_category = path_parts[-2] 
        print('True category: ', category_label_to_name[real_category])