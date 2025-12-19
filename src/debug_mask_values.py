import numpy as np
from torchvision.datasets import OxfordIIITPet

ds = OxfordIIITPet(root="data", split="trainval", target_types="segmentation", download=False)

_, mask = ds[0]
mask_np = np.array(mask)

print("Mask shape:", mask_np.shape, "dtype:", mask_np.dtype)
print("Unique values:", np.unique(mask_np))
