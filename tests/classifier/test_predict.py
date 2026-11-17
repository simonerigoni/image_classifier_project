import json
import sys

from src.classifier import predict


def test_parse_input_arguments_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["predict.py"])
    image_path, checkpoint_path, top_k, category_names, gpu = predict.parse_input_arguments()
    assert image_path.endswith("image_06760.jpg")
    assert checkpoint_path.endswith("checkpoint.pth")
    assert top_k == 5
    assert category_names.endswith("cat_to_name.json")
    assert gpu is True


def test_parse_input_arguments_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--image_path",
            "flower.jpg",
            "--checkpoint_path",
            "model.pth",
            "--top_k",
            "3",
            "--category_names",
            "names.json",
        ],
    )
    image_path, checkpoint_path, top_k, category_names, gpu = predict.parse_input_arguments()
    assert image_path == "flower.jpg"
    assert checkpoint_path == "model.pth"
    assert top_k == 3
    assert category_names == "names.json"


def test_get_mapping_label_name_categories(tmp_path):
    mapping = {"1": "pink primrose", "2": "hard-leaved pocket orchid"}
    path = tmp_path / "cat_to_name.json"
    path.write_text(json.dumps(mapping))
    assert predict.get_mapping_label_name_categories(str(path)) == mapping
