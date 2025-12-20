import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PetSegmentationDataset
from unet import UNet


def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: [B,1,H,W] (raw output)
    targets: [B,1,H,W] (0/1)
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
    # CPU-friendly defaults
    IMG_SIZE = 128
    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    full_ds = PetSegmentationDataset(root="data", split="trainval", img_size=IMG_SIZE)

    # Split 80/20 (train/val)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = UNet().to(device)

    # Loss = BCE + Dice (strong baseline)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coeff_from_logits(logits.detach(), masks).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)

                val_loss += loss.item()
                val_dice += dice_coeff_from_logits(logits, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_dice={train_dice:.4f} | val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

    # Save model
    torch.save(model.state_dict(), "runs/unet_pet_baseline.pth")
    print("Saved: runs/unet_pet_baseline.pth")


if __name__ == "__main__":
    main()
