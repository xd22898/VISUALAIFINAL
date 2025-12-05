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
from dataloader import DIV2K_SR16DatasetFull


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)

from DIPmodel import build_dip_model, optimize_dip_single_image

# DIP on x16 DIV2K LR images


# Run DIP on multiple images from DIV2K validation set
def run_dip_multiple(num_iter=3000, max_images=5):
    # Load the dataset
    dataset = DIV2K_SR16DatasetFull(
        lr_x8_dir=f"./DIV2K_valid_LR_x8",
        hr_dir=f"./DIV2K_valid_HR"
    )

    for index in range(min(max_images, len(dataset))):
        # Load the image pair
        lr, hr = dataset[index]

        if lr.shape[0] != 3:
            lr = lr.permute(2,0,1)
        if hr.shape[0] != 3:
            hr = hr.permute(2,0,1)

        lr_np = lr.detach().cpu().permute(1, 2, 0).numpy()
        hr_np = hr.detach().cpu().permute(1, 2, 0).numpy()

        print(f"\n=== DIP: Running on DIV2K image #{index} (scale x16) ===")

        # Run DIP on the selected image pair
        out_hr = optimize_dip_single_image(
            lr_np, hr_np, factor=16,
            num_iter=num_iter
        )

        print("HR:", hr_np.shape)
        print("OUT:", out_hr.shape)

        # If the output has 3 channels, convert it back to the correct format
        if out_hr.shape[0] == 3:  
            out_hr = np.transpose(out_hr, (1, 2, 0))
        
        print("HR:", hr_np.shape)
        print("OUT:", out_hr.shape)

        # Compute evaluation metrics
        psnr = compare_psnr(hr_np, out_hr)
        ssim = compare_ssim(hr_np, out_hr, channel_axis=2, data_range=1.0)
        lpips = calc_lpips(
            torch.tensor(out_hr).permute(2,0,1).unsqueeze(0).cuda(),
            torch.tensor(hr_np).permute(2,0,1).unsqueeze(0).cuda(),
            LPIPS_LOSS_FN
        )

        print("\n[DIP] Results for image #", index)
        print(f"  PSNR : {psnr:.4f}")
        print(f"  SSIM : {ssim:.4f}")
        print(f"  LPIPS: {lpips:.4f}")

        # Save metrics to a file for each image
        metrics_path = "dip_metrics_x16.txt"
        with open(metrics_path, "a") as f:
            f.write(
                f"Image {index} | PSNR={psnr:.4f} | SSIM={ssim:.4f} | LPIPS={lpips:.4f}\n"
            )
        print(f"[Saved metrics] → {metrics_path}")

        # Save super-resolution results as an image grid
        save_dir = "DIP_results_downscaled"
        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to images and save visual comparison
        save_full_image_comparison(
            x_lr_full=torch.tensor(lr_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE),
            x_pred_clamped=torch.clamp(torch.tensor(out_hr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE), 0, 1),
            x_hr_full=torch.tensor(hr_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE),
            epoch=num_iter,
            device=DEVICE,
            degradation_type="NONE",
            degradation_level=0,
            folder_name=save_dir
        )

        output_image_path = os.path.join(save_dir, f"final_{index}.png")
        print(f"[Saved DIP Super-Resolution Comparison] → {output_image_path}")



def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale', type=int, default=16)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--iters', type=int, default=3000)

    args = parser.parse_args()

    # Run DIP on the first 5 images
    run_dip_multiple(
        num_iter=args.iters,
        max_images=5 
    )



if __name__ == "__main__":
    main()
