import sys

import image_classifier_project as project


def test_parse_input_arguments_default_and_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["image_classifier_project.py"])
    assert project.parse_input_arguments().endswith("image_06760.jpg")

    monkeypatch.setattr(sys, "argv", ["image_classifier_project.py", "--image", "rose.jpg"])
    assert project.parse_input_arguments() == "rose.jpg"
