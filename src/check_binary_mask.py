# Verifying the binary mask

import matplotlib.pyplot as plt
from dataset import PetSegmentationDataset

ds = PetSegmentationDataset(root="data")
img, mask = ds[6]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(img.permute(1,2,0))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Binary Mask")
plt.imshow(mask.squeeze(), cmap="gray")
plt.axis("off")

plt.show()
