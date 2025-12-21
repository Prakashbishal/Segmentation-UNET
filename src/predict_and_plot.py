import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

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


def postprocess_mask(mask, kernel_size=5):
    """
    Robust post-processing for single-object segmentation
    mask: [H,W] float {0,1}
    """
    mask_uint8 = (mask * 255).astype(np.uint8)

    # --- Morphological closing (fill small holes) ---
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # --- Connected components ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    if num_labels <= 1:
        return (closed > 0).astype(np.float32)

    # Keep largest foreground component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    cleaned = np.zeros_like(mask_uint8)
    cleaned[labels == largest_label] = 255

    return (cleaned > 0).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    IMG_SIZE = 256
    ckpt_path = "runs/unet_pet_upgraded_best.pth"  # best checkpoint from upgraded training
    THRESHOLD = 0.5

    ds = PetSegmentationDataset(root="data", split="trainval", img_size=IMG_SIZE)

    idx = 0  # change to see different images
    img, gt_mask = ds[idx]

    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0)  # [H,W]
        pred_mask = (probs > THRESHOLD).float()
        # ---- DEBUG ----
        probs_np = probs.numpy()
        print("probs min/max:", probs_np.min(), probs_np.max())
        print("mean prob:", probs_np.mean())

        for t in [0.4, 0.5, 0.6, 0.7]:
            frac = (probs_np > t).mean()
            print(f"fraction > {t}: {frac:.4f}")
# --------------


    img_np = img.permute(1, 2, 0).numpy()
    gt_np = gt_mask.squeeze(0).numpy()

    pred_np = pred_mask.numpy()
    pred_np = postprocess_mask(pred_np)  # <-- cleanup step

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
    plt.title("Predicted Mask (Cleaned)")
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
