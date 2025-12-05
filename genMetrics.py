# genMetrics.py
from datetime import datetime
import os
import csv
import torch
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from PIL import Image
import torch.nn.functional as F


# LOGGER FOR METRICS

class MetricLogger:
    def __init__(self, save_dir, is_gan=False):
        """Creates metrics.csv inside the run directory"""
        self.filename = os.path.join(save_dir, "metrics.csv")
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            if is_gan:
                writer.writerow([
                    "epoch",
                    "discriminator_loss",
                    "generator_loss",
                    "val_psnr",
                    "val_ssim",
                    "val_lpips"
                ])
            else:
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_psnr",
                    "val_ssim",
                    "val_lpips"
                ])


    def log(self, epoch, train_loss, psnr, ssim, lpips_value):
        """Append one row to metrics.csv"""
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, psnr, ssim, lpips_value])

    def logGAN(self, epoch, discriminator_loss, gen_loss, psnr, ssim, lpips_value):
        """Append one row to metrics.csv"""
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, discriminator_loss, gen_loss, psnr, ssim, lpips_value])
       

    @staticmethod
    def create_run_folder(base="runs", prefix="ADMM_SR"):
        """Create a new run directory with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"{base}/{prefix}_{timestamp}"
        os.makedirs(folder, exist_ok=True)
        return folder
    

# METRICS 
def calc_psnr(img1, img2):
    if img1.shape != img2.shape:
        img2 = img2[:, :, :img1.shape[2], :img1.shape[3]]
    mse = F.mse_loss(img1.clamp(0,1), img2.clamp(0,1))
    return (10 * torch.log10(1 / (mse + 1e-8))).item()


def calc_ssim(img1, img2):
    if img1.shape != img2.shape:
        img2 = img2[:, :, :img1.shape[2], :img1.shape[3]]

    img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().numpy()

    total = 0
    for i in range(img1.shape[0]):
        total += sk_ssim(img1[i], img2[i], data_range=1.0, channel_axis=-1)
    return total / img1.shape[0]



def calc_lpips(img1, img2, lpips_model):
    if img1.shape != img2.shape:
        img2 = img2[:, :, :img1.shape[2], :img1.shape[3]]

    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    d = lpips_model(img1, img2)
    return d.mean().item()

# image save utilities
def tensor_to_pil(t):
    t = (t.clamp(0,1) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(t)


def save_qualitative_results(model, vis_loader, epoch, device, folder_name):
    model.eval()

    os.makedirs(folder_name, exist_ok=True)
    x_lr, x_hr = next(iter(vis_loader))
    x_lr, x_hr = x_lr.to(device), x_hr.to(device)

    with torch.no_grad():
        x_pred = torch.clamp(model(x_lr), 0, 1)
        x_bic = F.interpolate(
            x_lr, size=x_hr.shape[-2:], mode='bicubic', align_corners=False
        )

    hr = tensor_to_pil(x_hr[0])
    pred = tensor_to_pil(x_pred[0])
    bic = tensor_to_pil(x_bic[0])

    W = hr.width
    out = Image.new("RGB", (W * 3, W))

    out.paste(bic, (0, 0))
    out.paste(pred, (W, 0))
    out.paste(hr, (2 * W, 0))

    fname = f"{folder_name}/epoch_{epoch:03d}.png"
    out.save(fname)
    print(f"[Saved] {fname}")


def save_full_image_comparison(x_lr_full, x_pred_clamped, x_hr_full, epoch: int, device: str,  degradation_type: str, degradation_level: int, folder_name="qualitative_results_full", image_index= None):

    os.makedirs(folder_name, exist_ok=True)

    _, _, H_HR, W_HR = x_hr_full.shape
    x_bicubic = F.interpolate(x_lr_full, size=(H_HR, W_HR), mode='bicubic', align_corners=False)

    hr_img = tensor_to_pil(x_hr_full[0])
    pred_img = tensor_to_pil(x_pred_clamped[0])
    bicubic_img = tensor_to_pil(x_bicubic[0])
    
    total_width = bicubic_img.width + pred_img.width + hr_img.width
    total_height = hr_img.height 

    comparison_img = Image.new('RGB', (total_width, total_height))
    comparison_img.paste(bicubic_img, (0, 0))
    comparison_img.paste(pred_img, (bicubic_img.width, 0))
    comparison_img.paste(hr_img, (bicubic_img.width + pred_img.width, 0))

    unique_index = len(os.listdir(folder_name)) + 1  # Incremental index based on existing files
    out_path = f"{folder_name}/{degradation_type}_{degradation_level}_epoch_{epoch:03d}_img_{unique_index}.png"

    comparison_img.save(out_path)

    print(f"[Saved full-image qualitative] {out_path}")