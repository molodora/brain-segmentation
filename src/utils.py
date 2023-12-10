from pathlib import Path

import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data.datamodule import TCGADataset


def save_checkpoint(state, path):
    print("=> Saving checkpoint")
    torch.save(state, path)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def save_scripted_model(model, path):
    print("=> Saving final model")
    model_scripted = torch.jit.script(model.to("cpu"))
    model_scripted.save(path)


def get_loaders(
        dir,
        batch_size,
        trainsforms,
        num_workers=0,
        pin_memory=True
):
    dataset = TCGADataset(dir, trainsforms)
    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[0.8, 0.2],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cpu"):
    print("=> Checking metrics")

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    has_region = 0  # for custom dice score
    FP_pixels = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += 2 * (preds * y).sum() / ((preds + y).sum() + 1e-8)
            has_region += int(y.max())
            FP_pixels += ((preds == 1) & (y == 0)).sum()
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels:.2%}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"Custom dice score: {dice_score/has_region}")  # not including empty images
    print(f"Mean number of FP pixels: {FP_pixels/len(loader)}")
    model.train()


def save_preds_as_imgs(loader, model, folder="saved_images", device="cuda"):
    print("=> Saving predicted images")
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        img_path = Path(__file__).parent / folder
        torchvision.utils.save_image(
            preds, img_path / f"{idx}_pred.tif"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), img_path / f"{idx}_target.tif"
        )
    model.train()