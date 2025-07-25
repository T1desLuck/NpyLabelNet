import pytest
import torch
import numpy as np
import os
from PIL import Image
from auto_label import MiniCNN, load_classes, preprocess_image

def test_model_load():
    model = MiniCNN(num_classes=1000)
    assert isinstance(model, torch.nn.Module)
    assert model.fc2.out_features == 1000

def test_classes_load():
    classes = load_classes("classes.json")
    assert isinstance(classes, dict)
    assert "999" in classes
    assert classes["999"]["name"] == "неопределённый"

def test_preprocess_image():
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    img_tensor = preprocess_image(image)
    assert img_tensor.shape == (1, 3, 512, 512)
    assert img_tensor.dtype == torch.float32

def test_npy_output(tmp_path):
    model = MiniCNN(num_classes=5)
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, class_id = torch.max(probabilities, dim=1)
        class_id = class_id.item()
    npy_path = tmp_path / "test.npy"
    np.save(npy_path, np.array([class_id], dtype=np.int64))
    label = np.load(npy_path)
    assert label.shape == (1,)
    assert label.dtype == np.int64
    assert label[0] < 5
