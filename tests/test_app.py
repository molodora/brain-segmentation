import os
import sys
from pathlib import Path

import torch
import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.app import transform_image, app 


@pytest.fixture
def image():
    with open(Path(__file__).parent / "img.tif", 'rb') as f:
        image_bytes = f.read()
    return image_bytes


def test_loading_scripted_model():
    model_path = Path(__file__).parents[1] / "app" / "unet-scripted.pt"
    model = torch.jit.load(model_path)
    assert isinstance(model, torch.jit.ScriptModule)

def test_transform_image(image):
    tensor = transform_image(image)
    assert tensor.shape == torch.Size([1, 3, 256, 256])
    assert ((tensor >= 0) & (tensor <= 1)).all()

def test_model_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict", files = {"file": open(Path(__file__).parent / "img.tif", "rb")}
        )
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    with open(Path(__file__).parent /'test.png', 'wb') as f:
        f.write(response.content)