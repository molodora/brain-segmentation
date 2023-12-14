import os
from typing import Optional, Callable

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class TCGADataset(Dataset):
    """Dataset Class for dataloader.

    Attributes:
        image_dir (str): Directory with BMP images
        transform (Callable, optional): Various image transformations
            or augmentations, e.g. from torchvision.transform
    """
    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:

        self.image_dir = image_dir
        self.transform = transform

        filepaths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if '.tif' in file and 'mask' not in file:
                    filepaths.append(os.path.join(root, file))
        self.img_filepaths = filepaths

    def __len__(self) -> int:
        return len(self.img_filepaths)

    def __getitem__(self, idx: int):
        img_path = self.img_filepaths[idx]
        mask_path = self.img_filepaths[idx].replace(".tif", "_mask.tif")

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations['image']
            mask = augmentations['mask']

        return img, mask