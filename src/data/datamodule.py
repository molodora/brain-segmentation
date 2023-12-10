import os
from typing import Optional, Callable, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


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

if __name__ == "__main__":

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

    # data, target = next(iter(loader))
    # target = target.float().unsqueeze(1)
    # print(data.shape, target.shape)
    # print("data max", data.max(), "target max", target.max())

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        # print(data, data.shape)
        targets = targets.float().unsqueeze(1)
        # print(targets, targets.shape)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        preds = torch.randn((1, 1, 256, 256))
        loss = loss_fn(preds, targets)
        if loss < 0 or loss > 1:
            print()
            print(batch_idx)
            print("min", targets.min(), "max", targets.max())
            print(preds)
            print()
            print(loss)
            print()
            break
        # print(loss_fn(preds, targets))
        loop.set_postfix(loss=loss)