from src import config


def test_config_defaults():
    assert config.DEFAULT_NETWORK == "vgg16"
    assert config.DEFAULT_TOP_K == 5
    assert config.DEFAULT_HIDDEN_UNITS == 1024
    assert config.DATA_FOLDER == "data/"
    assert config.TRAIN_FOLDER.endswith("flowers/train")
    assert config.CHECKPOINT_FILENAME.endswith("checkpoint.pth")
