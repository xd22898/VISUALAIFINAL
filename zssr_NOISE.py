import os
import torch
import numpy as np
import lpips
import cv2
import csv

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from genMetrics import calc_lpips, save_full_image_comparison
from zssr import ZSSRNet, zssr_train_single, run_zssr_with_metrics
from degradations import add_gaussian_noise, add_gaussian_blur



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)


# Function to run ZSSR with different levels of noise and blur
def run_zssr_with_noise_and_blur(lr_img, hr_img, scale=8, noise_levels=[0, 15, 30, 50], blur_levels=[0, 3, 5, 7, 11], iters=20000):
    results = []

    # Test with noise levels first 
    for noise_level in noise_levels:
        print(f"\n=== Running ZSSR with noise level {noise_level} ===")

        # Add noise to the LR image
        noisy_lr_img = add_gaussian_noise(lr_img, noise_level)

        # Run ZSSR on noisy image and compute metrics
        out_hr, psnr, ssim, lpips_val = run_zssr_with_metrics(
            noisy_lr_img, hr_img, scale=scale, iters=iters
        )

        # Save the results for noise
        results.append([noise_level, 0, psnr, ssim, lpips_val])

    # Then test with blur levels 
    for blur_level in blur_levels:
        print(f"\n=== Running ZSSR with blur level {blur_level} ===")

        # Add blur to the LR image
        blurred_lr_img = add_gaussian_blur(lr_img, blur_level)

        # Run ZSSR on blurred image and compute metrics
        out_hr, psnr, ssim, lpips_val = run_zssr_with_metrics(
            blurred_lr_img, hr_img, scale=scale, iters=iters
        )

        # Save the results for blur
        results.append([0, blur_level, psnr, ssim, lpips_val])

    # Save results to a CSV file
    save_dir = "ZSSR_noise_blur_results"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "results_noise_blur_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Noise Level", "Blur Level", "PSNR", "SSIM", "LPIPS"])
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")




def main(lr_path, hr_path, scale=8, iters=20000):
    lr_img = cv2.imread(lr_path)[:, :, ::-1] / 255.0
    hr_img = cv2.imread(hr_path)[:, :, ::-1] / 255.0

    # Run ZSSR with different noise and blur levels
    run_zssr_with_noise_and_blur(lr_img, hr_img, scale=scale, noise_levels=[0, 15, 30, 50], blur_levels=[5, 7, 11], iters=iters)



if __name__ == "__main__":
    main(
        lr_path="DIV2K_valid_LR_x8/0803x8.png",
        hr_path="DIV2K_valid_HR/0803.png",
        scale=8,
        iters=20000   
    )
