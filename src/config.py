# Config
#
# python -m src.config


DATA_FOLDER = 'data/'
TRAIN_FOLDER = DATA_FOLDER + 'flowers/train'
TEST_FOLDER = DATA_FOLDER + 'flowers/valid'
VALID_FOLDER = DATA_FOLDER + 'flowers/test'
TEST_IMAGE = DATA_FOLDER + 'flowers/test/1/image_06760.jpg'
CHECKPOINT_FILENAME = DATA_FOLDER + 'checkpoint.pth'
CATEGORY_FILENAME = DATA_FOLDER + 'cat_to_name.json'
DEFAULT_NETWORK = 'vgg16'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_UNITS = 1024
DEFAULT_EPOCHS = 10
DEFAULT_TOP_K = 5
DEFAULT_GPU = True


if __name__ == "__main__":
    print("Config")
    print(f"{DATA_FOLDER = }")
    print(f"{TRAIN_FOLDER = }")
    print(f"{TEST_FOLDER = }")
    print(f"{VALID_FOLDER = }")
    print(f"{TEST_IMAGE = }")
    print(f"{CHECKPOINT_FILENAME = }")
    print(f"{CATEGORY_FILENAME = }")
    print(f"{DEFAULT_NETWORK = }")
    print(f"{DEFAULT_LEARNING_RATE = }")
    print(f"{DEFAULT_HIDDEN_UNITS = }")
    print(f"{DEFAULT_EPOCHS = }")
    print(f"{DEFAULT_TOP_K = }")
    print(f"{DEFAULT_GPU = }")
else:
    pass
