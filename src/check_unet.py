import torch
from unet import UNet

model = UNet()

x = torch.randn(1, 3, 128, 128)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
