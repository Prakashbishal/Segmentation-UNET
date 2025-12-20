import torch
import matplotlib.pyplot as plt

from dataset import PetSegmentationDataset
from unet import UNet


def overlay_mask_red(img, mask, alpha=0.45):
    """
    img: [H,W,3] in [0,1]
    mask: [H,W] in {0,1}
    """
    overlay = img.copy()
    # add red tint where mask==1
    overlay[..., 0] = overlay[..., 0] * (1 - alpha * mask) + alpha * mask
    overlay[..., 1] = overlay[..., 1] * (1 - alpha * mask)
    overlay[..., 2] = overlay[..., 2] * (1 - alpha * mask)
    return overlay


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    IMG_SIZE = 256
    ckpt_path = "runs/unet_pet_upgraded_best.pth"  # best checkpoint from upgraded training

    ds = PetSegmentationDataset(root="data", split="trainval", img_size=IMG_SIZE)

    idx = 0  # change to see different images
    img, gt_mask = ds[idx]

    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0)
        pred_mask = (probs > 0.5).float()

    img_np = img.permute(1, 2, 0).numpy()
    gt_np = gt_mask.squeeze(0).numpy()
    pred_np = pred_mask.numpy()

    over_pred = overlay_mask_red(img_np, pred_np)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_np, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Overlay (Prediction)")
    plt.imshow(over_pred)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
