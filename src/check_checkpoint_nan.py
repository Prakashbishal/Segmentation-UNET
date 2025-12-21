import torch
from unet import UNet

def main():
    ckpt = "runs/New Folder/unet_pet_upgraded_best.pth"
    device = "cpu"

    model = UNet()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    has_nan = False
    has_inf = False

    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print("NaN in:", name)
            has_nan = True
        if torch.isinf(p).any():
            print("Inf in:", name)
            has_inf = True

    print("Has NaN:", has_nan)
    print("Has Inf:", has_inf)

if __name__ == "__main__":
    main()
