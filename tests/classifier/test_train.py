import sys

import pytest

from src.classifier import train


def test_parse_input_arguments_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    network, learning_rate, hidden_units, epochs, gpu = train.parse_input_arguments()
    assert network == "vgg16"
    assert learning_rate == 0.001
    assert hidden_units == 1024
    assert epochs == 10
    assert gpu is True


def test_parse_input_arguments_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--network",
            "vgg13",
            "--learning_rate",
            "0.01",
            "--hidden_units",
            "512",
            "--epochs",
            "2",
        ],
    )
    network, learning_rate, hidden_units, epochs, gpu = train.parse_input_arguments()
    assert network == "vgg13"
    assert learning_rate == 0.01
    assert hidden_units == 512
    assert epochs == 2


def test_get_data_directories():
    train_dir, valid_dir, test_dir = train.get_data_directories()
    assert train_dir.endswith("flowers/train")
    assert valid_dir.endswith("flowers/test")  # VALID_FOLDER points at test in config
    assert test_dir.endswith("flowers/valid")


def test_get_number_of_classes_with_matching_dirs(tmp_path):
    for split, labels in (
        ("train", ["1", "2"]),
        ("valid", ["1", "2"]),
        ("test", ["1", "2"]),
    ):
        for label in labels:
            (tmp_path / split / label).mkdir(parents=True)

    assert (
        train.get_number_of_classes(
            str(tmp_path / "train"), str(tmp_path / "valid"), str(tmp_path / "test")
        )
        == 2
    )


def test_get_number_of_classes_mismatch_exits(tmp_path):
    (tmp_path / "train" / "1").mkdir(parents=True)
    (tmp_path / "valid" / "1").mkdir(parents=True)
    (tmp_path / "valid" / "2").mkdir(parents=True)
    (tmp_path / "test" / "1").mkdir(parents=True)
    with pytest.raises(SystemExit):
        train.get_number_of_classes(
            str(tmp_path / "train"), str(tmp_path / "valid"), str(tmp_path / "test")
        )
