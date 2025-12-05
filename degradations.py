import numpy as np
import cv2


# Gaussian Noise
def add_gaussian_noise(img, sigma):
    """
    img: HxWx3 float32 [0,1]
    sigma: noise std in pixel units
    """
    noise = np.random.randn(*img.shape) * (sigma / 255.0)
    return np.clip(img + noise, 0, 1)



# Gaussian Blur
def add_gaussian_blur(img, ksize):
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)



# Degradation dispatcher
def degrade(lr_np, degradation_type, level):
    if degradation_type == "noise":
        return add_gaussian_noise(lr_np, sigma=level)

    elif degradation_type == "blur":
        return add_gaussian_blur(lr_np, ksize=level)

    else:
        return lr_np

