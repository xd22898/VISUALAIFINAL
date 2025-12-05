import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lpips
import cv2, random

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from dataloader import DIV2K_FullImageDataset
from genMetrics import calc_lpips, save_full_image_comparison



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_LOSS_FN = lpips.LPIPS(net='vgg').to(DEVICE)



# ZSSR Network (8 layers, 64 channels)
class ZSSRNet(nn.Module):
    def __init__(self, num_layers=8, num_channels=64):
        super(ZSSRNet, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, num_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_channels, num_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(num_channels, 3, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)   # residual skip like original ZSSR

def build_scale_pyramid(lr_img, final_scale, num_steps=6):
    """
    Builds progressively larger target resolutions, from LR to final HR.
    Returns a list of scale factors like: [1.0, 1.2, 1.5, 2.0, ..., final]
    """
    scales = []
    current = 1.0
    step = (final_scale - 1.0) / num_steps

    for i in range(num_steps + 1):
        scales.append(current)
        current += step

    scales[-1] = final_scale  # ensure exact match
    return scales


def random_crop(img, size=64):
    H, W, _ = img.shape
    x = random.randint(0, W - size)
    y = random.randint(0, H - size)
    return img[y:y+size, x:x+size]

def zssr_train_single(
    lr_img,
    scale=8,
    patch_size=128,
    iters=20000,
    lr_rate=1e-5,
    num_scales=6,
    device="cuda"
):

    H0, W0 = lr_img.shape[:2]

    net = ZSSRNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)
    loss_fn = nn.L1Loss()

    # Build scale pyramid
    scales = np.linspace(1.0, scale, num_scales+1)

    # The LR image ALWAYS remains the source of HR-like patches
    lr_source = lr_img.copy()

    current_output = None  # updated after each scale

    for si, s in enumerate(scales):

        print(f"\n=== SCALE {si+1}/{len(scales)} — upscale {s:.3f} ===")

        iters_per_scale = iters // len(scales)

        # This is the 'inference target size' for this scale
        target_H = int(H0 * s)
        target_W = int(W0 * s)

        for i in range(iters_per_scale):

      
            # 1. Crop HR-like patch from LR image 
            hr_patch = random_crop(lr_source, patch_size) #this is the hr father

            # 2. Create LR_patch by downscaling
            # Random scale between 0.6–0.95 like true ZSSR
            s_down = np.random.uniform(0.60, 0.95)
            new_W = max(4, int(patch_size * s_down))
            new_H = max(4, int(patch_size * s_down))

            lr_small = cv2.resize(hr_patch, (new_W, new_H), interpolation=cv2.INTER_AREA)

            # mild blur is essential
            lr_small = cv2.GaussianBlur(lr_small, (3,3), sigmaX=0.5) #this is the lr son


            # 3. Upscale to ILR_patch
            ilr_patch = cv2.resize(lr_small, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

            # small noise injection (stabilizes training)
            ilr_patch += np.random.normal(0, 0.01, ilr_patch.shape).astype(np.float32)
            ilr_patch = np.clip(ilr_patch, 0, 1)

        
            # 4. Train CNN: ILR → HR_patch
            inp = torch.tensor(ilr_patch).permute(2,0,1).unsqueeze(0).float().to(device)
            tgt = torch.tensor(hr_patch).permute(2,0,1).unsqueeze(0).float().to(device)

            pred = net(inp)
            loss = loss_fn(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LR decay
            if (i % 3000 == 0) and (i > 0):
                for g in optimizer.param_groups:
                    g['lr'] *= 0.7
                print(f"LR decayed → {optimizer.param_groups[0]['lr']:.2e}")

            if i % 200 == 0:
                print(f"scale {si+1}, iter {i}/{iters_per_scale}, loss={loss.item():.4f}", end="\r")

      
        # 5. After training this scale, produce super resolution result at scale s
        up_lr = cv2.resize(lr_img, (target_W, target_H), interpolation=cv2.INTER_CUBIC)

        up_tensor = torch.tensor(up_lr).permute(2,0,1).unsqueeze(0).float().to(device)
        out_img = net(up_tensor).detach().cpu().squeeze().permute(1,2,0).numpy()
        out_img = np.clip(out_img, 0, 1)

        current_output = out_img  # final output for this scale

    print("\n=== TRUE ZSSR Training Complete ===")
    return current_output

def run_zssr_with_metrics(lr_img_np, hr_img_np, scale=8, iters=20000):

    print("\n=== Running Simplified Stable ZSSR with Metrics ===")


    # Run simplified ZSSR training
    out_hr = zssr_train_single(
        lr_img_np,
        scale=scale,
        patch_size=64,    # SAFE FOR DIV2K LR×8
        iters=iters,
        lr_rate=1e-5,
        num_scales=6,
        device=DEVICE
    )


    def center_crop_to_match(hr, out):
        H, W = out.shape[:2]
        h, w = hr.shape[:2]

        if h < H or w < W:
            raise ValueError("HR image is smaller than output, cannot crop.")

        y = (h - H) // 2
        x = (w - W) // 2
        return hr[y:y+H, x:x+W]

    # crop HR to match out_hr (DIV2K mismatch fix)
    hr_img_np = center_crop_to_match(hr_img_np, out_hr)


    # Compute Metrics
    psnr = compare_psnr(hr_img_np, out_hr)
    ssim = compare_ssim(hr_img_np, out_hr, channel_axis=2, data_range=1.0)
    lpips_val = calc_lpips(
        torch.tensor(out_hr).permute(2,0,1).unsqueeze(0).float().to(DEVICE),
        torch.tensor(hr_img_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE),
        LPIPS_LOSS_FN
    )

    print("\n[ZSSR Results]")
    print(f"  PSNR : {psnr:.4f}")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  LPIPS: {float(lpips_val):.4f}")

    save_full_image_comparison(
        torch.tensor(lr_img_np).permute(2,0,1).unsqueeze(0),
        torch.tensor(out_hr).permute(2,0,1).unsqueeze(0),
        torch.tensor(hr_img_np).permute(2,0,1).unsqueeze(0),
        epoch=iters,
        device=DEVICE,
        degradation_type="NONE",
        degradation_level=0,
        folder_name="ZSSR_results_simplified"
    )

    print("\n[Saved output] → ZSSR_results_simplified/\n")

    return out_hr, psnr, ssim, float(lpips_val)



def main(lr_path, hr_path, scale=8, iters=20000):
    lr_img = cv2.imread(lr_path)[:, :, ::-1] / 255.0
    hr_img = cv2.imread(hr_path)[:, :, ::-1] / 255.0

    out, psnr, ssim, lpips_val = run_zssr_with_metrics(
        lr_img, hr_img, scale=scale, iters=iters
    )

    print("\nFinished Simplified ZSSR.\n")
    print("Final Metrics:")
    print(f"  PSNR : {psnr:.4f}")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  LPIPS: {lpips_val:.4f}\n")

    return out


if __name__ == "__main__":
    main(
        lr_path="DIV2K_valid_LR_x8/0801x8.png",
        hr_path="DIV2K_valid_HR/0801.png",
        scale=8,
        iters=40000   
    )
