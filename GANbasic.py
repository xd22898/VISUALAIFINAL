import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch.nn as nn
import lpips 
from dataloader import DIV2K_SRDataset, DIV2K_FullImageDataset
from genMetrics import (
    calc_psnr, calc_ssim, calc_lpips,
    save_qualitative_results, save_full_image_comparison, MetricLogger
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)
SCALE_FACTOR = 8
BATCH_SIZE = 32
NUM_EPOCHS = 200


# Residual Block 
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# Generator
class SimpleSRGANGenerator(nn.Module):
    def __init__(self, num_res_blocks=8, scale_factor=8):
        super().__init__()

        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )

        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        # Conv + BN after residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        up_layers = []
        num_upsample = int(math.log2(scale_factor))
        for _ in range(num_upsample):
            up_layers += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*up_layers)

        # Final reconstruction
        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x1 = self.conv1(x)
        out = self.res_blocks(x1)
        out = x1 + self.conv2(out)
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.clamp(out, 0, 1)



# Discriminator
class SimpleSRGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_bn_lrelu(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, stride=s, padding=p),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        layers = [
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_bn_lrelu(64, 64, s=2),
            conv_bn_lrelu(64, 128, s=1),
            conv_bn_lrelu(128, 128, s=2),
            conv_bn_lrelu(128, 256, s=1),
            conv_bn_lrelu(256, 256, s=2),
            conv_bn_lrelu(256, 512, s=1),
            conv_bn_lrelu(512, 512, s=2)
        ]

        self.features = nn.Sequential(*layers)

        # 256×256 → 16×16 after downsampling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



# Loss functions
bce_logits = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()
l2 = nn.MSELoss()


def D_step(D, real, fake, optD):
    fake = torch.clamp(fake, 0, 1)

    pred_real = D(real)
    pred_fake = D(fake.detach())

    loss_real = bce_logits(pred_real, torch.ones_like(pred_real))
    loss_fake = bce_logits(pred_fake, torch.zeros_like(pred_fake))

    loss_D = 0.5 * (loss_real + loss_fake)

    optD.zero_grad()
    loss_D.backward()
    optD.step()

    return loss_D.item()


def G_step(G, D, lr, hr, optG):
    fake = G(lr)
    pred_fake = D(fake)

    #content loss
    loss_content = l1(fake, hr)
    #adversarial loss
    loss_adv = bce_logits(pred_fake, torch.ones_like(pred_fake))

    loss_G = loss_content + 0.001 * loss_adv  

    optG.zero_grad()
    loss_G.backward()
    optG.step()

    return loss_G.item(), fake.detach()




# Validation
def validate_epoch_gan(G, val_loader, device):
    G.eval()
    avg_psnr = avg_ssim = avg_lpips = 0.0

    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="GAN Validation", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            fake = torch.clamp(G(lr), 0, 1)

            avg_psnr += calc_psnr(fake, hr)
            avg_ssim += calc_ssim(fake, hr)
            avg_lpips += calc_lpips(fake, hr, LPIPS_LOSS_FN)

    n = len(val_loader)
    return avg_psnr / n, avg_ssim / n, avg_lpips / n



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

        if epoch % 20 == 0 or epoch == epochs:
            save_qualitative_results(G, vis_loader, epoch, device,
                                     folder_name="qualitative_results_GAN")

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
def evaluate_full_image_gan(G, scale=8, device=DEVICE, epochs=50):
    val_ds = DIV2K_FullImageDataset(
        lr_dir=f"./DIV2K_valid_LR_x{scale}",
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
        folder_name="qualitative_results_GAN_full"
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