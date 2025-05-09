# Image classifier project
#
# python image_classifier_project.py

import os
import argparse
import src.classifier.train as classifier_train
import src.classifier.predict as classifier_predict


from src.config import DATA_FOLDER, TRAIN_FOLDER, TEST_FOLDER, VALID_FOLDER, CATEGORY_FILENAME, CHECKPOINT_FILENAME, TEST_IMAGE, DEFAULT_NETWORK, DEFAULT_LEARNING_RATE, DEFAULT_HIDDEN_UNITS, DEFAULT_EPOCHS, DEFAULT_TOP_K, DEFAULT_GPU


def parse_input_arguments():
    '''
    Parse the command line arguments

    Args:
        None

    Returns:
        image (str): image path to be classified
    '''
    parser = argparse.ArgumentParser(description="Image Classfier")
    parser.add_argument('--image', type=str,
                        default=TEST_IMAGE, help='Image to classify')
    args = parser.parse_args()
    # print(args)
    return args.image


def load_classifier(checkpoint_filename=CHECKPOINT_FILENAME):
    '''
    Load the classifier. Train the network if needed

    Args:
        checkpoint_filename (str): checkpoint filename for loading the model

    Returns:
        model (object): model loaded from checkpoint
        category_label_to_name (dict): dictionary for mapping category label to name
    '''
    print('Check if checkpoint model present\n\tCheckpoint: {}'.format(
        checkpoint_filename))
    if os.path.isfile(checkpoint_filename) == False:
        print('Not present. Training the model')
        classifier_train.train(DEFAULT_NETWORK, DEFAULT_LEARNING_RATE,
                               DEFAULT_HIDDEN_UNITS, DEFAULT_EPOCHS, DEFAULT_GPU)
    else:
        print('Ok')

    model, category_label_to_name = classifier_predict.load_model(
        CATEGORY_FILENAME, checkpoint_filename)

    return model, category_label_to_name


def get_prediction(model, category_label_to_name, image_path):
    '''
    Get the prediction probability for the image given in input

    Args:
        model (object): model to make predictions
        category_label_to_name (dict): dictionary for mapping category label to name
        image_path: path of the image to classify

    Returns:
        probability_category (dict): dictonary with predicted category and probability
    '''
    top_probabilities, top_classes = classifier_predict.get_prediction(
        image_path, model, top_k_probabilities=DEFAULT_TOP_K)
    probability_category = []
    for i in range(len(top_probabilities)):
        probability_category.append(
            (top_probabilities[i], category_label_to_name[top_classes[i]]))
    return dict((category, probability) for probability, category in probability_category)


def get_number_of_classes():
    '''
    Return the number of classes used

    Args:
        None

    Returns:
        number_of_classes (int): number of classes used
    '''
    return classifier_train.get_number_of_classes(TRAIN_FOLDER, VALID_FOLDER, TEST_FOLDER)


def get_sample_from_training_dataset(category_label_to_name):
    '''
    Get a sample from the training dataset

    Args:
        category_label_to_name (dict): dictionary for mapping category label to name

    Returns:
        category_sample (dict): dictonary with category and image path
    '''
    directories = os.listdir(TRAIN_FOLDER)
    category_sample = []
    for directory in directories:
        files = os.listdir(TRAIN_FOLDER + '/' + directory)
        category_sample.append(
            (category_label_to_name[directory], TRAIN_FOLDER + '/' + directory + '/' + files[0]))
    return dict((category, sample) for category, sample in category_sample)


if __name__ == '__main__':
    print('Image classifier project')
    image_path = parse_input_arguments()
    model, category_label_to_name = load_classifier()

    print('Number of categories: {}'.format(get_number_of_classes()))
    # print(get_sample_from_training_dataset(category_label_to_name))

    category_probability = get_prediction(
        model, category_label_to_name, image_path)
    print(category_probability)

    if image_path == TEST_IMAGE:
        # we know the directory structure so the penultimate component of the path is the categoryh path/category/image.jpg
        path_parts = image_path.split('/')
        real_category = path_parts[-2]
        print('True category: ', category_label_to_name[real_category])
    else:
        pass
else:
    pass
