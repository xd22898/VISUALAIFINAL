import os
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from models.downsampler import Downsampler
import numpy as np
import torchvision.transforms.functional as TF


# Dataset for loading DIV2K LR-HR pairs for x8 super-resolution
# Extracts random patches while maintaining alignment.
class DIV2K_SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale_factor=8, hr_patch=256):
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        assert len(self.lr_files) == len(self.hr_files)

        self.scale = scale_factor
        self.hr_size = hr_patch
        self.lr_size = hr_patch // scale_factor

        self.to_tensor = transforms.ToTensor()

        # ntire2018 LR/HR bicubic pairs are aligned, but border padding exists → 2px safety margin
        self.margin = 2

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_files[idx]).convert('RGB')
        lr_img = Image.open(self.lr_files[idx]).convert('RGB')

        W_hr, H_hr = hr_img.size

        # Allowed HR crop region
        x_hr_min = self.margin * self.scale
        y_hr_min = self.margin * self.scale
        x_hr_max = W_hr - self.hr_size - self.margin * self.scale
        y_hr_max = H_hr - self.hr_size - self.margin * self.scale

        x_hr = random.randint(x_hr_min, x_hr_max)
        y_hr = random.randint(y_hr_min, y_hr_max)

        hr_patch = hr_img.crop((x_hr, y_hr, x_hr + self.hr_size, y_hr + self.hr_size))

        x_lr = x_hr // self.scale
        y_lr = y_hr // self.scale
        lr_patch = lr_img.crop((x_lr, y_lr, x_lr + self.lr_size, y_lr + self.lr_size))

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)


#FOR X16 GAN - use HR to create x16 LR directly because LR shape not always (16,16,3)
class DIV2K_SR16Dataset(Dataset):
    def __init__(self, lr_x8_dir, hr_dir, patch_size=256):
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.patch_size = patch_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):

        # Load HR
        hr = np.array(Image.open(self.hr_files[idx]).convert("RGB"), dtype=np.float32) / 255.0

        # Crop HR to divisible by 16
        H, W, _ = hr.shape
        H_new = (H // 16) * 16
        W_new = (W // 16) * 16
        hr = hr[:H_new, :W_new, :]

        # Force LR size exactly as expected.
        LR_H = H_new // 16
        LR_W = W_new // 16

        hr_t = torch.tensor(hr).permute(2,0,1).float()
        lr_t = TF.resize(hr_t, size=[LR_H, LR_W], interpolation=TF.InterpolationMode.BICUBIC)
        lr = lr_t.permute(1,2,0).numpy()

        # Patch sizes
        HR_PATCH = self.patch_size
        LR_PATCH = HR_PATCH // 16

        x = random.randint(0, H_new - HR_PATCH)
        y = random.randint(0, W_new - HR_PATCH)

        hr_patch = hr[x:x+HR_PATCH, y:y+HR_PATCH]
        lr_patch = lr[x//16:(x//16)+LR_PATCH, y//16:(y//16)+LR_PATCH]

        #print("LR shape:", lr_patch.shape, "HR shape:", hr_patch.shape)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)


# Full DIV2K x16 dataset loader (no patches)
class DIV2K_SR16DatasetFull(Dataset):
    def __init__(self, lr_x8_dir, hr_dir):
        self.lr_files = sorted(glob.glob(os.path.join(lr_x8_dir, "*.png")))
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        assert len(self.lr_files) == len(self.hr_files)

        self.to_tensor = transforms.ToTensor()

        self.down_x2 = Downsampler(
            n_planes=3,
            factor=2,
            kernel_type='lanczos2',
            preserve_size=False
        )

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):

        # Load HR and LR×8
        hr = np.array(Image.open(self.hr_files[idx]).convert("RGB"), dtype=np.float32) / 255.0
        lr8 = np.array(Image.open(self.lr_files[idx]).convert("RGB"), dtype=np.float32) / 255.0

        # Downscale LR8 → LR16
        lr16 = self.down_x2(
            torch.tensor(lr8).permute(2,0,1).unsqueeze(0)
        ).squeeze(0).permute(1,2,0).detach().cpu().numpy()

        # Now extract the exact LR shape from the downsampler
        LR_H, LR_W, _ = lr16.shape

        # HR must be exactly 16 × LR size
        HR_H = LR_H * 16
        HR_W = LR_W * 16

        hr = hr[:HR_H, :HR_W, :]

        return self.to_tensor(lr16), self.to_tensor(hr)

# Full DIV2K dataset loader (no patches)
class DIV2K_FullImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        lr = Image.open(self.lr_files[idx]).convert('RGB')
        return self.to_tensor(lr), self.to_tensor(hr)

