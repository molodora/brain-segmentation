from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from models.model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    save_scripted_model,
    get_loaders,
    check_accuracy,
    save_preds_as_imgs
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IN_CHANNELS = 3
OUT_CHANNELS = 1
PIN_MEMORY = True
LOAD_MODEL = True
IMG_DIR = Path(__file__).parents[1] / "data" / "raw"
MODEL_DIR = Path(__file__).parents[1] / "models"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    print("=> Training")
    model.train()
    
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)
        
        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        IMG_DIR,
        BATCH_SIZE,
        transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    checkpoint_path = MODEL_DIR / "checkpoint.pth.tar"
    if LOAD_MODEL:
        try:
            checkpoint = torch.load(checkpoint_path)
            load_checkpoint(checkpoint, model, optimizer)
        except FileNotFoundError:
            print(f"=> Failed to load checkpoint {checkpoint_path}")

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        if epoch % 3 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, checkpoint_path)
            check_accuracy(val_loader, model, device=DEVICE)

        if epoch == NUM_EPOCHS - 1:
            save_preds_as_imgs(val_loader, model)
            save_scripted_model(model, MODEL_DIR / "unet-scripted.pt")


if __name__ == "__main__":
    main()