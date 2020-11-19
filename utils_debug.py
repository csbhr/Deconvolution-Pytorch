import numpy as np
import math
import torch


def pre_process(img):
    img = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float().mul_(1. / 255)
    img = img.unsqueeze(0)
    return img


def post_process(img):
    img = img.mul(255.).clamp(0, 255).round()
    img = np.transpose(img[0].data.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


def PSNR(img1, img2):
    '''
    img1, img2: [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
