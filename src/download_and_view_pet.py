# Downloading a segmentation dataset automatically (no manual download)

import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ToTensor
from pathlib import Path

root = Path("data")

# target_types="segmentation" gives you pixel mask labels
ds = OxfordIIITPet(
    root=root,
    split="trainval",
    target_types="segmentation",
    download=True,
)

img, mask = ds[6]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask)
plt.axis("off")

plt.tight_layout()
plt.show()
