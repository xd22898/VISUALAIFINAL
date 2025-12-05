
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


# Build DIP network (skip architecture)
def build_dip_model(input_depth=16, pad="reflection"):
    net = get_net(
        input_depth=input_depth,
        NET_TYPE='skip',
        pad=pad,
        skip_n33d=32,
        skip_n33u=32,
        skip_n11=2,
        num_scales=6,
        upsample_mode='bilinear'
    ).cuda()
    return net


# Optimize DIP on a single image
def optimize_dip_single_image(
    lr_img, hr_img, factor,
    num_iter, lr=0.01,
    reg_noise_std=0.03, tv_weight=0.0
):

    H, W = hr_img.shape[:2]

    # DIP input noise
    net_input = get_noise(16, 'noise', (H, W)).cuda().detach()
    net = build_dip_model()

    # Downsampler to match LR → HR
    downsampler = Downsampler(
        n_planes=3,
        factor=factor,
        kernel_type="lanczos2",
        phase=0.5,
        preserve_size=True
    ).cuda()

    mse = nn.MSELoss().cuda()
    img_LR = np_to_torch(lr_img).permute(0, 3, 1, 2).cuda()

    net_input_saved = net_input.clone()
    noise = net_input.clone()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print(f"\n[DIP] Starting optimization for {num_iter} iterations...")

    for i in range(num_iter):
        optimizer.zero_grad()

        # Add noise regularization
        if reg_noise_std > 0:
            net_input = net_input_saved + noise.normal_() * reg_noise_std

        out_hr = net(net_input)
        out_lr = downsampler(out_hr)

        loss = mse(out_lr, img_LR)
        if tv_weight > 0:
            loss += tv_weight * tv_loss(out_hr)

        loss.backward()
        optimizer.step()

        if i % 5 == 0 or i == num_iter - 1:
            print(f" iter {i:05d}  loss={loss.item():.6f}", end="\r")

    print("")
    return torch_to_np(out_hr)



# Run DIP on ONE image from DIV2K
def run_dip_single(scale=8, index=0, num_iter=3000):
    """
    Selects ONE LR-HR image pair from DIV2K validation set.
    Applies DIP to super-resolve LR → HR.
    """

    dataset = DIV2K_FullImageDataset(
        lr_dir=f"./DIV2K_valid_LR_x{scale}",
        hr_dir=f"./DIV2K_valid_HR"
    )

    # Load one image pair
    lr, hr = dataset[index]

    if lr.shape[0] != 3:
        lr = lr.permute(2,0,1)
    if hr.shape[0] != 3:
        hr = hr.permute(2,0,1)

    lr_np = lr.permute(1,2,0).cpu().numpy()
    hr_np = hr.permute(1,2,0).cpu().numpy()

    print(f"\n=== DIP: Running on DIV2K image #{index} (scale x{scale}) ===")

    out_hr = optimize_dip_single_image(
        lr_np, hr_np, factor=scale,
        num_iter=num_iter
    )

    print("HR:", hr_np.shape)
    print("OUT:", out_hr.shape)

    if out_hr.shape[0] == 3:  
        out_hr = np.transpose(out_hr, (1, 2, 0))
    
    print("HR:", hr_np.shape)
    print("OUT:", out_hr.shape)

    # Compute metrics
    psnr = compare_psnr(hr_np, out_hr)
    ssim = compare_ssim(hr_np, out_hr, channel_axis=2, data_range = 1.0)
    lpips = calc_lpips(
        torch.tensor(out_hr).permute(2,0,1).unsqueeze(0).cuda(),
        torch.tensor(hr_np).permute(2,0,1).unsqueeze(0).cuda(),
        LPIPS_LOSS_FN
    )

    print("\n[DIP] Results for one image:")
    print(f"  PSNR : {psnr:.4f}")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  LPIPS: {lpips:.4f}")

   
    # save super resolution as image grid
    x_lr_full = torch.tensor(lr_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    x_hr_full = torch.tensor(hr_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    x_pred_full = torch.tensor(out_hr).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    x_pred_clamped = torch.clamp(x_pred_full, 0, 1)

    save_dir = "DIP_results"
    os.makedirs(save_dir, exist_ok=True)

    # Save visual comparison image
    save_full_image_comparison(
        x_lr_full=x_lr_full,
        x_pred_clamped=x_pred_clamped,
        x_hr_full=x_hr_full,
        epoch=5000,
        device=DEVICE,
        folder_name=save_dir
    )

    print(f"[Saved DIP Super-Resolution Comparison] → {save_dir}/final.png")

    return out_hr


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--iters', type=int, default=3000)

    args = parser.parse_args()

    run_dip_single(
        scale=args.scale,
        index=args.index,
        num_iter=args.iters
    )


if __name__ == "__main__":
    main()
