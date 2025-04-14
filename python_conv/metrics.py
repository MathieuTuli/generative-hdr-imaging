import torch


def psnr(img1: torch.Tensor, img2: torch.Tensor, pixel_max: float = 1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(pixel_max / torch.sqrt(mse))
