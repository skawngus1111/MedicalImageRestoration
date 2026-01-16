import os
import json
import pickle
import random

import torch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import imageio.v2 as imageio

def mkdir(p, is_file=False):
    if is_file:
        p, _ =  os.path.split(p)
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p))

class transformData:
    '''
    all-in-one medical image data
    '''

    def __init__(self, data_range=None):
        self.r = data_range
        self.data_range = {
            "CT": [-1024.0, 3071.0],
            "PET": [0.0, 20.0],
            "MRI": [0.0, 4095.0],
        }

        '''
        Abdomen CT image is truncated to [-160, 240] 
        reference: https://github.com/SSinyu/WGAN-VGG
        '''

        self.test_trucate_data_range = {
            "CT": [-160.0, 240.0],
            "PET": [0.0, 20.0],
            "MRI": [0.0, 4095.0],
        }

    def truncate(self, img, d_min, d_max):
        img[img > d_max] = d_max
        img[img < d_min] = d_min
        return img

    def truncate_test(self, img, modality):
        d_min, d_max = self.test_trucate_data_range[modality]
        img[img > d_max] = d_max
        img[img < d_min] = d_min
        return img

    def normalize(self, img, modality):
        d_min, d_max = self.data_range[modality]
        img = self.truncate(img, d_min, d_max)
        img = (img - d_min) / (d_max - d_min)

        return img

    def denormalize(self, img, modality):
        d_min, d_max = self.data_range[modality]
        img = img * (d_max - d_min) + d_min
        img = self.truncate(img, d_min, d_max)
        return img

    def random_crop(self, tensor, patch_size):
        """
        从给定的图像张量中随机裁剪大小为patch_size的patch。

        参数:
        tensor: 形状为[B, C, H, W]的图像张量。
        patch_size: 裁剪patch的大小，格式为(H, W)。

        返回:
        裁剪后的patch张量。
        """
        B, C, H, W = tensor.shape
        patch_h, patch_w = patch_size

        # 确保裁剪尺寸不大于原图像尺寸
        if patch_h > H or patch_w > W:
            raise ValueError("裁剪尺寸应小于原始图像尺寸")

        # 随机选择裁剪的起始点
        top = random.randint(0, H - patch_h)
        left = random.randint(0, W - patch_w)

        # 裁剪patch
        patches = tensor[:, :, top:top + patch_h, left:left + patch_w]
        return patches

    def random_rotate_flip(self, tensor):
        """
        对形状为[B, C, H, W]的图像张量执行随机旋转或翻转。

        参数:
        tensor: 形状为[B, C, H, W]的图像张量。

        返回:
        经过随机旋转或翻转的图像张量。
        """
        B, C, H, W = tensor.shape
        processed = torch.empty_like(tensor)

        for i in range(B):
            img = tensor[i]
            operation = torch.randint(0, 6, (1,)).item()

            if operation == 1:
                # 水平翻转
                img = torch.flip(img, [2])
            elif operation == 2:
                # 垂直翻转
                img = torch.flip(img, [1])
            elif operation == 3:
                # 旋转90度
                img = img.transpose(1, 2).flip(2)
            elif operation == 4:
                # 旋转180度
                img = img.flip(1).flip(2)
            elif operation == 5:
                # 旋转270度
                img = img.transpose(1, 2).flip(1)

            # 不做改变的情况下，operation == 0
            processed[i] = img

        return processed

    def _add_gaussian_noise(self, clean_patch, sigma):
        # 将sigma从[0,255]范围转换到[0,1]范围
        sigma = sigma / 255.0
        noise = torch.randn_like(clean_patch)
        noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 1)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patches, degrade_type=None):
        if degrade_type is None:
            degrade_type = random.randint(0, 2)

        B, C, H, W = clean_patches.shape
        degraded_patches = torch.empty_like(clean_patches)

        for i in range(B):
            degraded_patches[i], _ = self._degrade_by_type(clean_patches[i], degrade_type)

        return degraded_patches


class dataIO:
    def __init__(self):
        self.reader = {
            '.img': self.load_itk,
            '.gz': self.load_itk,
            '.nii': self.load_itk,
            '.bin': self.load_bin,
            '.txt': self.load_txt,
            '.json': self.load_json

        }
        self.writer = {
            '.img': self.save_itk,
            '.gz': self.save_itk,
            '.nii': self.save_itk,
            '.bin': self.save_bin,
            '.csv': self.save_csv,
            '.txt': self.save_txt,
            '.txt': self.save_json
        }

    def save_itk(self, data, path, use_int=False):
        if use_int:
            data = np.around(data)
        sitk.WriteImage(sitk.GetImageFromArray(data), path)

    def save_bin(self, data, path, use_int=False):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_itk(self, path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def load_bin(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def load_txt(self, path):
        with open(path, "r") as f:
            data = f.read()
        return data

    def save_json(self, data, path):
        with open(path, "w", encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, path):
        with open(path, encoding='utf8') as f:
            data = json.load(f)
        return data

    def save_csv(self, data_dict, path):
        result = pd.DataFrame({key: pd.Series(value) for key, value in data_dict.items()})
        result.to_csv(path)

    def save_txt(self, s, path):
        with open(path, 'w') as f:
            f.write(s)

    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex

    def load(self, path):
        ex = self.getFileEX(path)
        return self.reader[ex](path)

    def save(self, data, path):
        mkdir(path, is_file=True)
        ex = self.getFileEX(path)
        return self.writer[ex](data, path)


def to_uint8_vis(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    x: (H,W) numpy float
    vmin/vmax: display range
    return: uint8 (H,W) in [0,255]
    """
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-12)
    x = (x * 255.0).round().astype(np.uint8)
    return x

# def save_png_vis(tensor, path: str, modality: str, tfm, use_test_range=True):
#     """
#     tensor: torch Tensor [1,1,H,W] (or [H,W]) on GPU/CPU
#     tfm: transformData instance
#     """
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#
#     x = tensor.detach().float().cpu().numpy()
#     x = x.squeeze()  # -> (H,W)
#
#     # AMIR transformData 기준: test_trucate_data_range 사용 권장 (CT: [-160,240], PET: [0,20], MRI: [0,4095])
#     if use_test_range:
#         vmin, vmax = tfm.test_trucate_data_range[modality]
#     else:
#         vmin, vmax = tfm.data_range[modality]
#
#     x_u8 = to_uint8_vis(x, vmin, vmax)
#     imageio.imwrite(path, x_u8)

def save_png_vis_final(tensor, path: str, modality: str, tfm):
    import numpy as np
    import imageio.v2 as imageio
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    x = tensor.detach().float().cpu().numpy().squeeze()

    if modality == "CT":
        # 논문용 고정 window
        vmin, vmax = -160.0, 240.0

    elif modality == "PET":
        # PET은 percentile + cap
        x_pos = x[x > 0]
        if x_pos.size > 10:
            vmax = np.percentile(x_pos, 99.5)
            vmax = min(vmax, 6.0)   # 핵심
        else:
            vmax = 1.0
        vmin = 0.0

    elif modality == "MRI":
        # MRI는 percentile window가 정석
        vmin = np.percentile(x, 1.0)
        vmax = np.percentile(x, 99.0)

    else:
        vmin, vmax = float(x.min()), float(x.max())

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-12)
    x = (x * 255.0).round().astype(np.uint8)

    imageio.imwrite(path, x)