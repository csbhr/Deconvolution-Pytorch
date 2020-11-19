import numpy as np
import math
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F


def pre_process(img):
    img = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float().mul_(1. / 255)
    img = img.unsqueeze(0)
    return img


def post_process(img):
    img = img.mul(255.).clamp(0, 255).round()
    img = np.transpose(img[0].detach().data.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


def pre_process_kernel(kernel):
    kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()
    return kernel


def post_process_kernel(kernel):
    mi, ma = torch.min(kernel), torch.max(kernel)
    kernel = (kernel - mi) / (ma - mi)
    kernel = kernel.mul(255.).clamp(0, 255).round()
    kernel = torch.cat([kernel, kernel, kernel], dim=1)
    kernel = np.transpose(kernel[0].detach().data.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return kernel


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


def get_lr_blurdown(img_gt, kernel, scale, dowsammple_method='bicubic'):
    img_gt = np.array(img_gt).astype('float32')
    gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    gt_tensor = F.pad(gt_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(gt_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]
    blur_tensor = blur_tensor.clamp(0, 255).round()
    blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('float32')

    if dowsammple_method == 'nearest':
        blurdown = blur[::scale, ::scale, :]
    elif dowsammple_method == 'bicubic':
        blurdown = cv2.resize(blur, dsize=(0, 0), fx=1. / scale, fy=1. / scale, interpolation=cv2.INTER_CUBIC)
    else:
        raise Exception('Not support downsample method "{}"'.format(dowsammple_method))

    blurdown = blurdown.astype('uint8')

    return blurdown
