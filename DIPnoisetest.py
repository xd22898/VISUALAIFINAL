
import numpy as np
import torch
import torch.nn as nn
import lpips 
import os

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from models import get_net
from utils.common_utils import get_noise 
from models.downsampler import Downsampler
from utils.sr_utils import np_to_torch, torch_to_np, tv_loss

from genMetrics import calc_lpips, save_full_image_comparison
from dataloader import DIV2K_FullImageDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)

from DIPmodel import build_dip_model, optimize_dip_single_image

# Run DIP on ONE image from DIV2K with various degradations

def run_final_degradation_tests_DIP(scale=8, index=0, num_iter=3000, device="cuda"):

    from degradations import add_gaussian_noise, add_gaussian_blur
    import csv

    save_dir = "final_degradation_results_DIP"
    os.makedirs(save_dir, exist_ok=True)


    # Load one LR/HR image pair
    dataset = DIV2K_FullImageDataset(
        lr_dir=f"./DIV2K_valid_LR_x{scale}",
        hr_dir=f"./DIV2K_valid_HR"
    )

    lr, hr = dataset[index]

    lr_np = lr.permute(1,2,0).cpu().numpy().astype(np.float32)
    hr_np = hr.permute(1,2,0).cpu().numpy().astype(np.float32)

    noise_levels = [0, 15, 30, 50]
    blur_levels  = [0, 3, 5, 7, 11]

    results = []


    # Helper: run DIP on degraded LR image and compute metrics
    def run_case(kind, level, degraded_np):

        degraded_np = degraded_np.astype(np.float32)  # <<< FIX

        print(f"\n[DIP] Running {kind}={level}")

        out_hr = optimize_dip_single_image(
            degraded_np, hr_np,
            factor=scale,
            num_iter=num_iter
        )
        if out_hr.shape[0] == 3:  
            out_hr = np.transpose(out_hr, (1, 2, 0))
    

        # metrics
        psnr = compare_psnr(hr_np, out_hr)
        ssim = compare_ssim(hr_np, out_hr, channel_axis=2, data_range=1.0)

        lp = calc_lpips(
            torch.tensor(out_hr).permute(2,0,1).unsqueeze(0).float().to(device),
            torch.tensor(hr_np).permute(2,0,1).unsqueeze(0).float().to(device),
            LPIPS_LOSS_FN
        )

        # Save comparison image 
        degraded_t = torch.tensor(degraded_np).permute(2,0,1).unsqueeze(0).float().to(device)
        pred_t     = torch.tensor(out_hr)     .permute(2,0,1).unsqueeze(0).float().to(device)
        hr_t       = torch.tensor(hr_np)      .permute(2,0,1).unsqueeze(0).float().to(device)

        save_full_image_comparison(
            degraded_t, pred_t, hr_t,
            epoch=5000,
            device=device,
            degradation_type=kind,
            degradation_level=level,
            folder_name=save_dir
        )

        results.append([kind, level, psnr, ssim, float(lp)])

 
    # Gaussian Noise Tests
    for sigma in noise_levels:
        degraded = add_gaussian_noise(lr_np, sigma).astype(np.float32)
        run_case("noise", sigma, degraded)

    # Gaussian Blur Tests
    for k in blur_levels:
        degraded = add_gaussian_blur(lr_np, k).astype(np.float32)
        run_case("blur", k, degraded)


    # Save CSV Summary
    csv_path = os.path.join(save_dir, "results_DIP.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Level", "PSNR", "SSIM", "LPIPS"])
        writer.writerows(results)

    print("\n=== DIP Degradation Tests Complete ===")
    print(f"Saved results to: {save_dir}")

    return results


def main():
    run_final_degradation_tests_DIP(scale=8, index=0, num_iter=5000)

if __name__ == "__main__":
    main()
