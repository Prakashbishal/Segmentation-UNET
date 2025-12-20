import torch
import matplotlib.pyplot as plt

from dataset import PetSegmentationDataset
from unet import UNet


def overlay_mask(img, mask, alpha=0.5):
    """
    img: [H,W,3] numpy in [0,1]
    mask: [H,W] numpy in {0,1}
    returns overlay image
    """
    overlay = img.copy()
    overlay[..., 1] = overlay[..., 1] * (1 - alpha * mask)  # reduce green where mask=1
    overlay[..., 0] = overlay[..., 0] * (1 - alpha * mask)  # reduce red where mask=1
    # blue channel remains -> gives a bluish highlight region
    return overlay


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load one sample
    ds = PetSegmentationDataset(root="data", split="trainval", img_size=128)
    img, gt_mask = ds[0]  # try different indices later

    # Load trained model
    model = UNet().to(device)
    model.load_state_dict(torch.load("runs/unet_pet_baseline.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))  # [1,1,H,W]
        probs = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0)  # [H,W]
        pred_mask = (probs > 0.5).float()

    # Convert to numpy for plotting
    img_np = img.permute(1, 2, 0).numpy()
    gt_np = gt_mask.squeeze(0).numpy()
    pred_np = pred_mask.numpy()

    over_gt = overlay_mask(img_np, gt_np)
    over_pred = overlay_mask(img_np, pred_np)

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
