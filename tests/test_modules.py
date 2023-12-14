import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.datamodule import TCGADataset
from src.models.model import UNet


@pytest.fixture
def loader():
    transforms = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )

    dataset = TCGADataset('data/raw', transforms)
    loader = DataLoader(dataset, batch_size=1)
    return loader


def test_img_shape(loader):
    data, _ = next(iter(loader))
    assert data.shape == torch.Size([1, 3, 256, 256])

def test_img_values(loader):
    data, _ = next(iter(loader))
    assert ((data >= 0) & (data <= 1)).all()

def test_mask_shape(loader):
    _, target = next(iter(loader))
    assert target.shape == torch.Size([1, 256, 256])

def test_mask_values(loader):
    _, target = next(iter(loader))
    assert ((target == 0) | (target == 1)).all()

def test_forward_tensor_shape():
    x = torch.randn((4, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    assert preds.shape == torch.Size([4, 1, 256, 256])
