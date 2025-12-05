import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import lpips 
from dataloader import DIV2K_SR16Dataset, DIV2K_SR16DatasetFull
from genMetrics import (
    calc_psnr, calc_ssim, calc_lpips,
    save_qualitative_results, save_full_image_comparison,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)
SCALE_FACTOR = 16
BATCH_SIZE = 32
NUM_EPOCHS = 200


from genMetrics import MetricLogger

from GANbasic import (
   SimpleSRGANGenerator, 
   SimpleSRGANDiscriminator, 
   validate_epoch_gan, 
   train_simple_sr_gan,
   D_step, G_step)




# Training loop using x16 downsampled aligned patches
def train_simple_sr_gan(scale, batch_size, epochs, device=DEVICE):

    train_ds = DIV2K_SR16Dataset(
        lr_x8_dir=f"./DIV2K_train_LR_x8",
        hr_dir="./DIV2K_train_HR",
        patch_size = 256
    )
    val_ds = DIV2K_SR16Dataset(
        lr_x8_dir=f"./DIV2K_valid_LR_x8",
        hr_dir="./DIV2K_valid_HR",
        patch_size = 256
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4)
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

        avg_g = g_sum / count
        avg_d = d_sum / count

        avg_psnr, avg_ssim, avg_lpips = validate_epoch_gan(G, val_loader, device)

        if epoch % 30 == 0 or epoch == epochs:
            save_qualitative_results(G, vis_loader, epoch, device,
                                     folder_name="qualitative_results_GAN_downscaled")

        logger.logGAN(epoch, avg_g, avg_d, avg_psnr, avg_ssim, avg_lpips)

        print(f"\n[GAN] Epoch {epoch}/{epochs}")
        print(f" G Loss: {avg_g:.6f}")
        print(f" D Loss: {avg_d:.6f}")
        print(f" PSNR:   {avg_psnr:.4f}")
        print(f" SSIM:   {avg_ssim:.4f}")
        print(f" LPIPS:  {avg_lpips:.4f}")
        print("-" * 60)

    return G, D



# Final full-image evaluation
def evaluate_full_image_gan(G, scale=16, device=DEVICE, epochs=200):
    val_ds = DIV2K_SR16DatasetFull(
        lr_x8_dir=f"./DIV2K_valid_LR_x8",
        hr_dir="./DIV2K_valid_HR"
    )
    loader = DataLoader(val_ds, batch_size=1)

    print("\n" + "=" * 60)
    print("[GAN] FINAL FULL-IMAGE EVAL")

    lr_full, hr_full = next(iter(loader))
    lr_full, hr_full = lr_full.to(device), hr_full.to(device)

    with torch.no_grad():
        pred = torch.clamp(G(lr_full), 0, 1)

    psnr = calc_psnr(pred, hr_full)
    ssim = calc_ssim(pred, hr_full)
    lpips = calc_lpips(pred, hr_full, LPIPS_LOSS_FN)

    save_full_image_comparison(
        lr_full, pred, hr_full, epochs, device, "none", 0,
        folder_name="qualitative_results_GAN_full_downscaled"
    )

    print(f"PSNR:  {psnr:.4f} dB")
    print(f"SSIM:  {ssim:.4f}")
    print(f"LPIPS: {lpips:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    G, D = train_simple_sr_gan(
        scale=SCALE_FACTOR,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        device=DEVICE
    )

    evaluate_full_image_gan(
        G,
        scale=SCALE_FACTOR,
        device=DEVICE,
        epochs=NUM_EPOCHS
    )