import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

def compute_measure(x, pred, data_range):
    pred_psnr = compute_PSNR(x, pred, data_range)
    pred_ssim = compute_SSIM(x, pred, data_range)
    pred_rmse = compute_RMSE(x, pred)
    return pred_psnr, pred_ssim, pred_rmse


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range=1.0, window_size=11, window=None):
    """
    img1, img2: [B,C,H,W] torch.Tensor (CUDA 가능)
    data_range: 1.0 (0~1 정규화) 또는 255.0 등
    """
    # SSIM은 AMP 영향 받지 않게 float32로 고정하는게 가장 안전
    with torch.cuda.amp.autocast(enabled=False):
        img1 = img1.float()
        img2 = img2.float()

        device = img1.device
        dtype  = img1.dtype  # float32

        if window is None:
            # 예시: 기존 코드가 window를 생성하던 로직을 그대로 두되,
            # 반드시 float32로 만들고 아래에서 dtype/device를 맞추면 됩니다.
            window = create_window(window_size, channel=img1.size(1))  # 기존 함수 사용
        window = window.to(device=device, dtype=dtype)

        # 이제 conv2d 입력/weight dtype이 동일
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

        mu1_sq  = mu1.pow(2)
        mu2_sq  = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

        C1 = (0.01 * float(data_range)) ** 2
        C2 = (0.03 * float(data_range)) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window