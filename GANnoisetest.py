import os
import csv
import numpy as np
import torchvision.utils as vutils
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import lpips 
from dataloader import DIV2K_SRDataset, DIV2K_FullImageDataset
from genMetrics import (
    calc_psnr, calc_ssim, calc_lpips,save_full_image_comparison)
from torch.utils.data import DataLoader
from degradations import add_gaussian_blur, add_gaussian_noise
from genMetrics import save_qualitative_results, MetricLogger

from GANbasic import (
   SimpleSRGANGenerator, 
   SimpleSRGANDiscriminator, 
   train_simple_sr_gan,
   D_step, G_step)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)
SCALE_FACTOR = 8
BATCH_SIZE = 32
NUM_EPOCHS = 200


# Training loop
def train_simple_sr_gan(scale, batch_size, epochs, device=DEVICE):

    train_ds = DIV2K_SRDataset(
        lr_dir=f"./DIV2K_train_LR_x{scale}",
        hr_dir="./DIV2K_train_HR",
        scale_factor=scale
    )
    val_ds = DIV2K_SRDataset(
        lr_dir=f"./DIV2K_valid_LR_x{scale}",
        hr_dir="./DIV2K_valid_HR",
        scale_factor=scale
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    vis_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    G = SimpleSRGANGenerator(scale_factor=scale).to(device)
    D = SimpleSRGANDiscriminator().to(device)

    optG = Adam(G.parameters(), lr=1e-4)
    optD = Adam(D.parameters(), lr=1e-5)

    # Logging setup
    run_dir = MetricLogger.create_run_folder(prefix=f"GAN_x{SCALE_FACTOR}")
    logger = MetricLogger(run_dir, is_gan=True)


    # Epoch loop
    for epoch in range(1, epochs + 1):
        G.train()
        D.train()

        g_sum = d_sum = count = 0

        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch}"):
            lr, hr = lr.to(device), hr.to(device)

            fake = G(lr)
            d_loss = D_step(D, hr, fake, optD)
            g_loss, _ = G_step(G, D, lr, hr, optG)

            g_sum += g_loss
            d_sum += d_loss
            count += 1
        if epoch % 5 == 0 or epoch == epochs:
            save_qualitative_results(G, vis_loader, epoch, device,
                                     folder_name="qualitative_results_GAN_noise15")


    return G, D


# Evaluate one degraded LR image
def eval_single(G, lr, hr, device):
    G.eval()
    with torch.no_grad():
        pred = torch.clamp(G(lr), 0, 1)

    psnr = calc_psnr(pred, hr)
    ssim = calc_ssim(pred, hr)
    lp = calc_lpips(pred, hr, LPIPS_LOSS_FN)

    return pred, psnr, ssim, lp



# Final image degradation evaluation
def run_final_degradation_tests(G, scale=8, device="cuda"):

    save_dir = "final_degradation_results_GAN"
    os.makedirs(save_dir, exist_ok=True)

    # Load FULL IMAGE validation set (just 1 img)
    val_ds = DIV2K_FullImageDataset(
        lr_dir=f"./DIV2K_valid_LR_x{scale}",
        hr_dir="./DIV2K_valid_HR",
    )
    lr_full, hr_full = val_ds[4]
    hr_full = hr_full.unsqueeze(0).to(device)  # shape (1,3,H,W)

    # Convert LR to numpy for degradations
    lr_np = lr_full.permute(1,2,0).cpu().numpy()

    noise_levels = [0, 15, 30, 50]
    blur_levels = [0, 3, 5, 7, 11]

    results = []

    # Helper to run and save results
    def run_case(kind, level, degraded_img_np):

        degraded_tensor = (
            torch.from_numpy(degraded_img_np)
            .permute(2,0,1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        pred, psnr, ssim, lp = eval_single(G, degraded_tensor, hr_full, device)

        # save output image
        save_full_image_comparison(
            degraded_tensor, pred, hr_full, epoch=200, device=device,
            degradation_type=kind, degradation_level=level, folder_name=save_dir
        )
        results.append([kind, level, psnr, ssim, float(lp)])


  
    # Gaussian Noise
    for sigma in noise_levels:
        degraded = add_gaussian_noise(lr_np, sigma)
        run_case("noise", sigma, degraded)
        print(f"[NOISE σ={sigma}] done")


    # Gaussian Blur
    for k in blur_levels:
        degraded = add_gaussian_blur(lr_np, k)
        run_case("blur", k, degraded)
        print(f"[BLUR k={k}] done")


 
    # Save CSV summary
    with open(os.path.join(save_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Level", "PSNR", "SSIM", "LPIPS"])
        writer.writerows(results)

    print("\n=== FINAL DEGRADATION TEST COMPLETE ===")
    print("Results saved to:", save_dir)

    return results


if __name__ == "__main__":
    G, D = train_simple_sr_gan(
        scale=SCALE_FACTOR,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        device=DEVICE
    )
    run_final_degradation_tests(
        G,
        scale=SCALE_FACTOR,
        device=DEVICE
    )