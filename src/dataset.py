# Converting this mask to binary

import torch
import numpy as np
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import torchvision.transforms.functional as F

class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="trainval", img_size=128):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=False
        )

        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # image -> tensor
        img = self.img_transform(img)

        # mask -> resize with NEAREST (important), then numpy
        mask = F.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        mask = np.array(mask).astype(np.int64)

        # Oxford-IIIT Pet is a trimap.
        # - {0: background, 1: pet, 2: border}
        # - {1: pet, 2: background, 3: border}
        # We map "pet OR border" to 1, background to 0.

        unique = np.unique(mask)

        if set(unique).issubset({0, 1, 2}):
            # pet=1, border=2, background=0
            bin_mask = (mask == 1) | (mask == 2)
        elif set(unique).issubset({1, 2, 3}):
            # pet=1, border=3, background=2
            bin_mask = (mask == 1) | (mask == 3)
        else:
            # fallback: treat largest non-zero region as foreground (rare)
            bin_mask = mask > mask.min()

        bin_mask = bin_mask.astype(np.float32)

        # to tensor: [1, H, W]
        bin_mask = torch.from_numpy(bin_mask).unsqueeze(0)

        return img, bin_mask
