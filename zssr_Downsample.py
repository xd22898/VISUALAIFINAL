import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lpips
import cv2
import random

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from dataloader import DIV2K_SR16DatasetFull
from genMetrics import calc_lpips, save_full_image_comparison
from zssr import ZSSRNet, random_crop, zssr_train_single


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)


# Function to run ZSSR with evaluation metrics
def run_zssr_with_metrics(lr_img_np, hr_img_np, scale=16, iters=20000):
    print("\n=== Running Simplified Stable ZSSR with Metrics ===")
    out_hr = zssr_train_single(
        lr_img_np,
        scale=16,
        patch_size=16,  #safe for downsampled images  
        iters=iters,
        lr_rate=1e-5,
        num_scales=6,
        device=DEVICE
    )

    # Crop HR to match out_hr (DIV2K mismatch fix)
    def center_crop_to_match(hr, out):
        H, W = out.shape[:2]
        h, w = hr.shape[:2]

        if h < H or w < W:
            raise ValueError("HR image is smaller than output, cannot crop.")

        y = (h - H) // 2
        x = (w - W) // 2
        return hr[y:y + H, x:x + W]

    hr_img_np = center_crop_to_match(hr_img_np, out_hr)

    # Compute Metrics
    psnr = compare_psnr(hr_img_np, out_hr)
    ssim = compare_ssim(hr_img_np, out_hr, channel_axis=2, data_range=1.0)
    lpips_val = calc_lpips(
        torch.tensor(out_hr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE),
        torch.tensor(hr_img_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE),
        LPIPS_LOSS_FN
    )

    print("\n[ZSSR Results]")
    print(f"  PSNR : {psnr:.4f}")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  LPIPS: {float(lpips_val):.4f}")

    # Save LR / OUT / HR image comparison
    save_full_image_comparison(
        torch.tensor(lr_img_np).permute(2, 0, 1).unsqueeze(0),
        torch.tensor(out_hr).permute(2, 0, 1).unsqueeze(0),
        torch.tensor(hr_img_np).permute(2, 0, 1).unsqueeze(0),
        epoch=iters,
        device=DEVICE,
        degradation_type="NONE",
        degradation_level=0,
        folder_name="ZSSR_results_downsample"
    )

    print("\n[Saved output] → ZSSR_results_simplified/\n")

    return out_hr, psnr, ssim, float(lpips_val)


# Running the ZSSR model with dataset integration
def run_zssr_with_loader(lr_path, hr_path, scale=16, iters=20000):
    # Initialize dataset loader (for LR×16 images)
    dataset = DIV2K_SR16DatasetFull(
        lr_x8_dir=f"./DIV2K_valid_LR_x8",
        hr_dir=f"./DIV2K_valid_HR"
    )

    # Test with first image in the dataset
    lr_img, hr_img = dataset[0]

    # Convert tensors to numpy (in case they are in tensor format)
    lr_img_np = lr_img.permute(1, 2, 0).numpy()
    hr_img_np = hr_img.permute(1, 2, 0).numpy()

    print(f"Running ZSSR on image {0}")

    # Perform ZSSR training with metrics evaluation
    out, psnr, ssim, lpips_val = run_zssr_with_metrics(
        lr_img_np, hr_img_np, scale=scale, iters=iters
    )

    return out, psnr, ssim, lpips_val


if __name__ == "__main__":
    out, psnr, ssim, lpips_val = run_zssr_with_loader(
        lr_path="DIV2K_valid_LR_x8/0801x8.png",
        hr_path="DIV2K_valid_HR/0801.png",
        scale=8,
        iters=40000  
    )

    print("Finished ZSSR.")
    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips_val:.4f}")
