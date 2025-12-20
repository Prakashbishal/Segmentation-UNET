import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import PetSegmentationDataset
from unet import UNet


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits:  [B,1,H,W] raw logits
    targets: [B,1,H,W] float {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)

    return dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()


def main():
    seed_everything(42)

    IMG_SIZE = 256
    BATCH_SIZE = 16     
    EPOCHS = 20
    LR = 1e-3

    NUM_WORKERS = 4     
    PIN_MEMORY = True

    os.makedirs("runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    full_ds = PetSegmentationDataset(root="data", split="trainval", img_size=IMG_SIZE)

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = UNet().to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_dice = dice_coeff_from_logits(logits.detach(), masks).item()
            train_loss += loss.item()
            train_dice += batch_dice

            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]")
            for imgs, masks in pbar:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with autocast(enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)

                batch_dice = dice_coeff_from_logits(logits, masks).item()
                val_loss += loss.item()
                val_dice += batch_dice
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_dice={train_dice:.4f} | val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "runs/unet_pet_upgraded_best.pth")
            print(f"Saved best: runs/unet_pet_upgraded_best.pth (val_dice={best_val_dice:.4f})")

    # Save final checkpoint
    torch.save(model.state_dict(), "runs/unet_pet_upgraded_final.pth")
    print("Saved final: runs/unet_pet_upgraded_final.pth")


if __name__ == "__main__":
    main()
