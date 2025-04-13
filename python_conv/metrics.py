import numpy as np


def psnr(img1: np.ndarray, img2: np.ndarray, pixel_max: float = 1.0):
    mse = np.mean(np.power((img1 - img2), 2))
    if mse == 0:
        return float("inf")
    return 20 * np.log10(pixel_max / np.sqrt(mse))
